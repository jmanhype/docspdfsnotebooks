import sys
import os
import pyperclip
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QScrollArea
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import Qt
from peewee import SqliteDatabase, Model, CharField
from flask import Flask, render_template
from openai import OpenAI
from semantic_router import Route, RouteLayer
from semantic_router.encoders import OpenAIEncoder

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
os.environ["OPENAI_API_KEY"] = "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"
encoder = OpenAIEncoder()
app = QApplication(sys.argv)
app.setStyle('Fusion')

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

    def generate_prompt(self):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Create an art prompt that reflects a juxtaposition of a vivid, colorful landscape with elements of surrealism and mystery. Include a transformative aspect where the conventional bounds of nature and perception are blended together, similar to revealing a hidden layer beneath the surface of reality."}]
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
