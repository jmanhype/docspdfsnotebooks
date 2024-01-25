import sys
import os
import pyperclip
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QScrollArea, QTextEdit  
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt
from peewee import SqliteDatabase, Model, CharField
from flask import Flask, render_template
from openai import OpenAI
from semantic_router import Route, RouteLayer
from semantic_router.encoders import OpenAIEncoder
import spacy

# Set up database with Peewee
db = SqliteDatabase('artprompts.db')

class ArtPrompt(Model):
    content = CharField()

    class Meta:
        database = db

db.connect()
db.create_tables([ArtPrompt])

# Initialize the OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

app = QApplication(sys.argv)
app.setStyle('Fusion')

# Load spaCy for natural language processing
nlp = spacy.load("en_core_web_sm")

# Color scheme and aesthetics
dark_palette = QPalette()
dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
dark_palette.setColor(QPalette.WindowText, Qt.white)
dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
dark_palette.setColor(QPalette.ToolTipText, Qt.white)
dark_palette.setColor(QPalette.Text, Qt.white)
dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
dark_palette.setColor(QPalette.ButtonText, Qt.white)
dark_palette.setColor(QPalette.BrightText, Qt.red)
dark_palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
dark_palette.setColor(QPalette.HighlightedText, Qt.black)
app.setPalette(dark_palette)

flask_app = Flask(__name__)

# Define Semantic Router routes
abstract_art_route = Route(
    name="abstract_art",
    utterances=[
        "create an abstract painting",
        "use vibrant colors in an abstract style",
        "depict emotion through abstract art",
        "explore abstract expressionism"
    ],
)

landscape_route = Route(
    name="landscape",
    utterances=[
        "paint a serene landscape",
        "capture the beauty of nature in your art",
        "create a landscape with mountains and rivers",
        "illustrate a cityscape at night"
    ],
)

portrait_route = Route(
    name="portrait",
    utterances=[
        "draw a realistic portrait",
        "create a portrait with unique facial expressions",
        "paint a portrait using only primary colors",
        "sketch a self-portrait in pencil"
    ],
)

still_life_route = Route(
    name="still_life",
    utterances=[
        "arrange a still life with fruits",
        "paint a still life scene in oil",
        "create a monochromatic still life drawing",
        "explore light and shadow in still life"
    ],
)

surrealism_route = Route(
    name="surrealism",
    utterances=[
        "create a surreal landscape",
        "paint a scene with dream-like elements",
        "explore the theme of surrealism in your art",
        "combine reality and fantasy in your artwork"
    ],
)

photography_route = Route(
    name="photography",
    utterances=[
        "capture a landscape in black and white photography",
        "explore portrait photography with natural light",
        "create a photo series telling a story",
        "experiment with long exposure photography"
    ],
)

digital_art_route = Route(
    name="digital_art",
    utterances=[
        "design a digital character concept",
        "create a digital landscape illustration",
        "experiment with digital abstract art",
        "develop a series of digital patterns and textures"
    ],
)

historical_styles_route = Route(
    name="historical_styles",
    utterances=[
        "create an artwork inspired by the Renaissance",
        "explore Impressionist painting techniques",
        "recreate a famous artwork in the style of Cubism",
        "study and apply techniques from Baroque art"
    ],
)

# Initialize Semantic Router's Route Layer
encoder = OpenAIEncoder()  # Assumes you have an OpenAI API key set in your environment variables
routes = [
    abstract_art_route, landscape_route, portrait_route, still_life_route,
    surrealism_route, photography_route, digital_art_route, historical_styles_route
]  # Add all defined routes here
route_layer = RouteLayer(encoder=encoder, routes=routes)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.user_input_text = QTextEdit(self) 
        self.setWindowTitle('Art Prompt Generator')
        self.setGeometry(100, 100, 800, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2D2D2D;
            }
            QPushButton {
                background-color: #8E2DC5;
                border-radius: 5px;
                padding: 15px;
                font-size: 16px;
                color: #FFFFFF;
            }
            QPushButton:hover {
                background-color: #9D30E5;
            }
            QLabel {
                border-radius: 5px;
            }
            QScrollArea {
                border: none;
            }
        """)

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        self.generate_button = QPushButton('Generate Art Prompt')
        self.generate_button.clicked.connect(self.generate_prompt)
        layout.addWidget(self.generate_button)

        # Add a text box for user input
        self.user_input_text = QTextEdit(self)
        self.user_input_text.setPlaceholderText("Enter your art prompt here...")
        layout.addWidget(self.user_input_text)

        # Add a new button for enhancing prompts
        self.enhance_button = QPushButton('Enhance Prompt')
        self.enhance_button.clicked.connect(self.enhance_prompt)
        layout.addWidget(self.enhance_button)

        # Add a button for copying to clipboard
        self.copy_button = QPushButton('Copy to Clipboard')
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        layout.addWidget(self.copy_button)

        self.prompt_label = QLabel("Your art prompt will appear here.")
        self.prompt_label.setWordWrap(True)
        font = QFont("Arial", 18, QFont.Bold)
        self.prompt_label.setFont(font)
        self.prompt_label.setStyleSheet("color: #32CD32; padding: 20px;")
        self.prompt_label.setAlignment(Qt.AlignCenter)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.prompt_label)
        layout.addWidget(scroll_area)

        # Reduced Prompt Label
        self.reduced_prompt_label = QLabel("Reduced prompt will appear here.")
        self.reduced_prompt_label.setWordWrap(True)
        font = QFont("Arial", 14)
        self.reduced_prompt_label.setFont(font)
        self.reduced_prompt_label.setStyleSheet("color: #32CD32; padding: 20px;")
        self.reduced_prompt_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.reduced_prompt_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
    def extract_key_elements(self, user_input):
        """
        Extract key elements (nouns and verbs) from the user input.
        """
        doc = nlp(user_input)
        key_elements = [token.text for token in doc if token.pos_ in ["NOUN", "VERB"]]
        return key_elements

    def formulate_query(self, key_elements):
        """
        Formulate a creative query for the OpenAI API using key elements.
        """
        base_query = "Create a detailed and imaginative art prompt based on the following elements: "
        query = base_query + ", ".join(key_elements)
        return query

    def get_enhanced_prompt(self, query):
        """
        Use the OpenAI API to generate a creative enhancement based on the query.
        """
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": query}]
            )
            enhanced_prompt = response.choices[0].message.content
            return enhanced_prompt
        except Exception as e:
            print(f"Error while generating enhanced prompt: {e}")
            return None
        
    def generate_prompt(self):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Create an art prompt consisting of an adjective, a noun, a verb, an art style, and a camera feature"}]
            )
            prompt_content = response.choices[0].message.content
            # Process the prompt through Semantic Router
            route_choice = route_layer(prompt_content)
            
            if route_choice.name:
                # Generate enhancements using GPT-4
                enhancements = self.generate_enhancements(route_choice.name)
                formatted_prompt = enhancements if enhancements else prompt_content
                self.prompt_label.setText(formatted_prompt)  # Update the label with the formatted prompt
                
                # Reduce the prompt to one or two words separated by commas
                reduced_prompt = self.reduce_prompt(formatted_prompt)
                self.reduced_prompt_label.setText(reduced_prompt)  # Update the reduced prompt label
            else:
                self.prompt_label.setText(prompt_content)
        except Exception as e:
            self.prompt_label.setText(f"An error occurred: {e}")
            print(f"An error occurred: {e}")

    def enhance_prompt(self):
        try:
            user_input = self.user_input_text.toPlainText()  # Get user input from the text box
            if not user_input:
                self.prompt_label.setText("Please enter a valid art prompt.")
                return

            # Generate an art prompt based on the user's input
            response_user_input = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_input}]
            )
            prompt_user_input = response_user_input.choices[0].message.content

            # Extract key elements (adjective, noun, verb, art style, camera feature) from the user input
            key_elements = self.extract_key_elements(prompt_user_input)

            if not key_elements:
                self.prompt_label.setText("Please enter a valid art prompt.")
                return

            # Formulate a query for creative expansion
            query = self.formulate_query(key_elements)

            # Get the enhanced prompt using the OpenAI API with shorter parameters
            enhanced_prompt = self.get_shortened_enhanced_prompt(query)

            if enhanced_prompt:
                # Combine the original prompt and the enhanced prompt
                combined_prompt = f"Original Prompt: \"{prompt_user_input}\"\n\nEnhanced Prompt: \"{enhanced_prompt}\""

                self.prompt_label.setText(combined_prompt)  # Update the label with both prompts

                # Reduce the combined prompt to one or two words separated by commas
                reduced_prompt = self.reduce_prompt(combined_prompt)
                self.reduced_prompt_label.setText(reduced_prompt)  # Update the reduced prompt label
            else:
                self.prompt_label.setText(f"Original Prompt: \"{prompt_user_input}\"")
        except Exception as e:
            self.prompt_label.setText(f"An error occurred: {e}")
            print(f"An error occurred: {e}")

    def get_shortened_enhanced_prompt(self, query):
        """
        Use the OpenAI API to generate a creative enhancement based on the query with shorter parameters.
        """
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": query}],
                max_tokens=50  # Limit the length of the generated response
            )
            enhanced_prompt = response.choices[0].message.content
            return enhanced_prompt
        except Exception as e:
            print(f"Error while generating enhanced prompt: {e}")
        return None




    def generate_enhancements(self, route_name):
        try:
            # Use GPT-4 to generate enhancements based on the route
            gpt4_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Enhance an art prompt for the {route_name} route"}]
            )
            enhancements = gpt4_response.choices[0].message.content
            return enhancements
        except Exception as e:
            print(f"An error occurred while generating enhancements: {e}")
            return None

    def reduce_prompt(self, formatted_prompt):
        try:
            # Use GPT-4 to summarize the prompt into one or two words separated by commas
            gpt4_response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": f"Summarize the art prompt: '{formatted_prompt}'"}]
            )
            reduced_prompt = gpt4_response.choices[0].message.content

            # Ensure that the reduced prompt consists of one or two words separated by commas
            words = reduced_prompt.split()
            if len(words) > 2:
                reduced_prompt = ", ".join(words[:2])  # Take the first two words

            return reduced_prompt
        except Exception as e:
            print(f"An error occurred while reducing the prompt: {e}")
            return formatted_prompt
        
    def copy_to_clipboard(self):
        text_to_copy = self.prompt_label.text()
        pyperclip.copy(text_to_copy)  # Use pyperclip to copy the text to clipboard
        print("Text copied to clipboard:", text_to_copy)  # Provide feedback to the user

@flask_app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
