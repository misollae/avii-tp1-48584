import os
import io
import json
import librosa
import tempfile
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import pipeline
from streamlit_mic_recorder import mic_recorder
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration, AutoTokenizer, AutoModelWithLMHead

system_base_instructions = "You'll be working a bit like a Dungeon Master, your task will be to create an interactable story played along with the user. \
                  This story should be primarily character interaction-driven, with the player's actions triggering reactions from the characters in the world, while still mantaining a plot flow.\
                  These characters should have defining traits and personalities that facilitate clear and distinct player interactions, while remaining themed with the story.\
                  They can also mantain a inventory (for example, a 'Bartender' may contain a stock of drinks) that can be updated and changed through player interactions.\
                  \
                  Your response should be in JSON format, formatted according to these rules:"

def init_openAI():
  ''' Inicializa e retorna um cliente OpenAI, usando as credenciais guardadas num ficheiro .dotenv.

    Returns:
        OpenAI: Cliente OpenAI
    '''
  load_dotenv()
  return OpenAI()

client = init_openAI()

def ensure_reply(temperature, messages):
  ''' Procura garantir uma resposta da API OpenAI Chat.

    Args:
        temperature (float): A temperatura a ser usada na geração da resposta.
        messages (list): Uma lista de mensagens a serem enviadas para a API.

    Returns:
        dict: Os dados da resposta da API.
    '''
  attempts = 0
  success = False

  print(messages[1])
  while attempts < 5 and not success:
    try:
        response = client.chat.completions.create(model='gpt-3.5-turbo', messages=messages, temperature=temperature)
        print(response)
        data     = json.loads(response.choices[0].message.content)
        success  = True
        return data  
    except Exception as e:
        attempts += 1

def get_character_response(character_name, user_message, temperature):
  '''Obtém a resposta do personagem para a mensagem do utilizador.

    Args:
        character_name (str): O nome do personagem.
        user_message (dict): A mensagem do utilizador.
        temperature (float): A temperatura a ser usada na geração da resposta.
    '''
  # Obter a informação da personagem
  character_entry = next((character for character in st.session_state['characters'] if character["Name"] == character_name), None)
  
  # Preparar as mensagens de sistema e utilizador
  character_instructions = f"You'll be working as a character in a story, your task will be to interact with the user.\
                            Here is the information about your character: {character_entry}.\
                            The user may interact with your inventory items, but you must ensure you own the item in question, otherwise just deny the request.\
                            Respond with a JSON with the following format (use double quotes):\
                            - 'Response': What your character says, try to prioritize dialogue over narration, describing character actions is also fine.\
                            - 'Character_Inventory': Optional parameter. If the interaction with the user affects your character inventory (ex: player acquires a character item so the item is removed and the money is added), return the character inventory. Make sure you include all items, not just the updated one.\
                            - 'Player_Inventory': If the interaction with the user affects the player inventory (ex: the player paid for something the character owns so the player gains the item but loses the money), return the updated inventory. Make sure you include all items, not just the updated one." 
  character_message = {'role': 'system', 'content': character_instructions}
  
  # Obtém resposta
  data = ensure_reply(temperature, [character_message, user_message])

  # Guarda os valores na sessão
  if 'Character_Inventory' in data:
    next((character.update({'Inventory': data['Character_Inventory']}) for character in st.session_state['characters'] if character['Name'] == character_name), None)
  if 'Player_Inventory' in data:
    st.session_state['player_inventory'] = data['Player_Inventory']

  print(data)
  st.session_state['chat'].append({"Speaker": character_name, "Message": data['Response']})

def handle_voice_input(temperature):
    '''Lida com a entrada de voz do utilizador.

    Args:
        temperature (float): A temperatura a ser usada na geração da resposta.
    '''
    if st.session_state.audio_recorder_output:
        # Obter a informação sobre o sinal de áudio
        audio         = st.session_state.audio_recorder_output
        
        # Alterar o sample rate
        y, sr      = librosa.load(io.BytesIO(audio["bytes"]), sr=audio["sample_rate"], mono=True)
        audio_data = librosa.resample(y, orig_sr=sr, target_sr=16000)

        # Tokenizar e processar
        model         = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
        processor     = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")
        inputs        = processor(audio_data, sampling_rate=16000, return_tensors="pt")
        generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

        # Obter a transcrição e passá-la como interação
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
        if transcription[0] != "":
          handle_user_input(temperature, transcription[0])

def handle_user_input(temperature, user_input):
  ''' Lida com o input do utilizador - trata um passo de interação.

  Args:
      temperature (float): A temperatura a ser usada na geração da resposta.
      user_input (str): Input do utilizador.
  '''
  # Guardar a interação
  st.session_state['chat'].append({"Speaker": "Player", "Message": user_input})

  # Preparar as mensagens
  information = {'last_interactions': st.session_state['chat'], 'characters': st.session_state['characters'], 'player_inventory': st.session_state['player_inventory']}
  json_format = "The format of the JSON is conditional (use double quotes on the attributes):\
                    1. If the player's action triggers a response or action from one of the characters in the story (or is directed towards one), based on the description or conversation continuation:\
                     - 'Responder': The content of this attribute should be the character's name - this will be used to redirect the player interaction to a AI entity with that character's personality.\
                    2. Otherwise, if the player's actions doesn't trigger any dialogue, only triggering the continuation of the story narration:\
                     - 'Story': This attribute should contain the next step of the story. Make sure this attribute does not contain direct dialogue from the characters, only narration or descriptions.\
                     - 'Player_Inventory': If the interaction with the user affects the player inventory, return the updated inventory. Make sure you include all items, not just the updated one." 
  system_message = {'role': 'system', 'content': system_base_instructions + json_format}
  user_message   = {'role': 'user', 'content': f"Continue the interaction considering:\n {information}"}

  # Obter a resposta
  data = ensure_reply(temperature, [system_message, user_message])

  # "Reencaminhar" para uma personagem, se for indicado
  if 'Responder' in data:
    get_character_response(data['Responder'], user_message, temperature)
  elif 'Story' in data:
    if 'Player_Inventory' in data:
      st.session_state['player_inventory'] = data['Player_Inventory']
    st.session_state['chat'].append({"Speaker": "Narrator", "Message": data['Story']})    
  st.rerun()

def handle_step(temperature):
  '''Lida com um passo da história, mostrando a informação atual e preparando a receção de um novo input.

    Args:
        temperature (float): A temperatura a ser usada na geração da resposta.
    '''
  st.markdown("<h5 style='margin-bottom: -20px'>💭 Player: </h5>", unsafe_allow_html=True)
  col1, col2 = st.columns([14,1])
  with col1:
    user_input = st.text_area("Your response:", placeholder="Your response (an action, a sentence...)", key="user_input_area", label_visibility="collapsed")
    formatted_inventory = ', '.join(f"{value} {key}" for key, value in st.session_state['player_inventory'].items())
    st.markdown(f"<p style='margin-top: -10px'><b>➥ Player Inventory: </b> {formatted_inventory}</p>", unsafe_allow_html=True)
  with col2:
    st.write("#\n###")
    mic_recorder(key="audio_recorder", start_prompt="🎙️", stop_prompt="❌", just_once=False, use_container_width=False, format="wav", callback=handle_voice_input, args=(temperature,), kwargs={})
  
  if st.button("Submit"):
    if user_input:
      handle_user_input(temperature, user_input)
  
# Métodos de Display
def display_characters():
  '''Exibe os personagens na interface do Streamlit.'''
  st.markdown("<h4 style='margin-bottom: 20px; margin-top: 0px'>Characters 👥</h4>", unsafe_allow_html=True)
  characters = st.session_state.get('characters', [])
  num_characters = len(characters)
  for index, character in enumerate(characters):
    st.write(f"<p style='margin-bottom: 2px; font-size: 22px;'><b>{character['Emoji']} {character['Name']}</b></p>", unsafe_allow_html=True)
    st.write(f"<p style='margin-bottom: 2px'>{character['About']}</p>", unsafe_allow_html=True)
    if character['Inventory']:
      formatted_inventory = ', '.join(f"{value} {key}" for key, value in character['Inventory'].items())

      st.write(f"<p style='margin-bottom: 2px'><b>Inventory:</b> {formatted_inventory}</p>", unsafe_allow_html=True)
    if index < num_characters - 1: 
        st.write("---")
  
  st.write(f"<p style='margin-top: 30px'></p>", unsafe_allow_html=True)
    
def display_story():
  '''Exibe a história na interface do Streamlit.'''
  last_step = st.session_state['chat'][-1]
  emoji = next((character['Emoji'] for character in st.session_state['characters'] if character['Name'] == last_step['Speaker']), "📖")
  st.markdown(f"<h5 style='margin-bottom: -30px; margin-top: 20px'> {emoji} {last_step['Speaker']}: </h5>", unsafe_allow_html=True)
  st.write(f"\n{last_step['Message']}")

def download_state():
  '''Permite que o usuário baixe o estado atual da história.'''
  state = json.dumps({'last_interactions': st.session_state['chat'], 'characters': st.session_state['characters']}, indent=4)
  json_bytes = state.encode('utf-8')
  st.download_button(label="Download current state", data=json_bytes, file_name='interactive_story.json', mime='application/json')

# MÉTODOS DE INICIALIZAÇÃO DA HISTÓRIA 
def start_story(temperature, prompt = None):
    '''Inicia a história com um tema fornecido.

    Args:
        temperature (float): A temperatura a ser usada na geração da resposta.
        prompt (str, optional): O tema fornecido para iniciar a história. Pode ser None. Defaults to None.
    '''
    # Preparação das mensagens
    json_format = "- 'Story' (required): Main output, the first step of the story.\
                   - 'Characters' (required): Characters on the story. 3-4 entries each containing 'Name', 'About', 'Inventory', 'Emoji', according to the explanation at the start.\
                  An example of a good characters would be a Name: Seller, About: A fruit seller who loves to discuss football and hates dogs, having a inventory of {'oranges' : 7, 'bananas' : 4}, for example.\
                  An ideal 'about' description should contain information about the character's personality, accent or dialect they speak in (should be easy to translate to text), conversation topics they like or dislike... Anything that produces interesting interactions.\
                  An ideal 'inventory' should contain items that can be interacted with and even obtained by the player.\
                  The 'emoji' should be the name of one emoji that fits the character.\
                   - 'Player_Inventory' (required): Starting player inventory, (example: {'knife' : 1, 'coins' : 100}), any items owned by the player. It should start with any items you find fitting (money, weapons...). Start with at least 1 item."
    system_message = {'role': 'system', 'content': system_base_instructions + json_format}
    message = 'Start the story.' + (f" Use the following theme: {prompt}" if prompt is not None else "")
    response = client.chat.completions.create(model='gpt-3.5-turbo', temperature=temperature, messages=[system_message, {'role': 'user', 'content': message}])
    
    # Obtenção das respostas
    data     = json.loads(response.choices[0].message.content)

    if (len(data['Characters']) >= 3):
      st.session_state['chat']              = [{"Speaker": "Narrator", "Message": data['Story']}]
      st.session_state['characters']        = data['Characters']
      st.session_state['player_inventory']  = data['Player_Inventory']
    st.rerun()

def get_prompt_from_image(image_file):
    '''Guarda os dados de uma imagem num ficheiro temporário, obtendo, via pipeline, uma descrição da imagem.

    Args:
        image_file (BytesIO): O ficheiro de imagem fornecido.

    Returns:
        str: A descrição extraída da imagem.
    '''
    temp_file   = tempfile.NamedTemporaryFile(suffix=f".{image_file.name.split('.')[-1]}", delete=False)    
    temp_file.write(image_file.getvalue())    

    image2text  = pipeline(task="image-to-text", model="Salesforce/blip-image-captioning-base")
    description = image2text(temp_file.name, max_new_tokens=600)[0]['generated_text']
    temp_file.close()
    os.unlink(temp_file.name)
    return description

def setup_page():  
  '''Configura a página inicial.'''
  st.set_page_config(layout="centered")

  # Título da página
  st.markdown("<h3 style='text-align: center; margin-bottom: -35px''>Initialize Theme ✒️</h3><br>", unsafe_allow_html=True)
  temperature = st.slider('Temperature:', min_value=0.0, max_value=2.0, value=1.0, step=0.01)
  
  # Opção 1 :: Enviar uma imagem como setting da história
  st.markdown("<h6 style='margin-bottom: -25px'; margin-top: 100px'>OPTION 1 ⤳ Use image description:</h6>", unsafe_allow_html=True)
  image_file = st.file_uploader("Use image description:", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
  if image_file is not None:
    with st.spinner('Processing...'):
      start_story(temperature, prompt = get_prompt_from_image(image_file))

  # Opção 2 :: Escrever diretamente o tema
  st.markdown("<h6 style='margin-bottom: -25px; margin-top: 12px'>OPTION 2 ⤳ Manually insert the theme:</h6>", unsafe_allow_html=True)
  text_input = st.text_area("Theme...", placeholder="Write story theme...", label_visibility="collapsed")
  if text_input:
    with st.spinner('Processing...'):
      start_story(temperature, prompt = text_input.strip())

  # Opção 3 :: Botão para randomizar o prompt
  st.markdown("<h6 style='margin-bottom: -25px; margin-top: 12px'>OPTION 3 ⤳ Start with a randomly generated theme:</h6>", unsafe_allow_html=True)
  if st.button("Randomize"):
    with st.spinner('Processing...'):
      start_story(temperature)


def main_page():
  '''Exibe a página principal.'''
  st.set_page_config(layout="wide")
  col1, colA, col2, colB, col3 = st.columns([155, 10, 330, 10, 100])

  with col1.container(height=550):
    display_characters()

  with col2:
    temperature = st.slider('Temperature:', min_value=0.0, max_value=2.0, value=1.0, step=0.01)  
    display_story()
    handle_step(temperature)

  with col3:
    download_state()
    if st.button("Quit story"):
      st.session_state.clear()
      st.rerun()

def main():
  '''Função principal para executar o aplicativo.'''
  if 'chat' not in st.session_state:
    setup_page()
  else:
    main_page()

if __name__ == "__main__":
    main()