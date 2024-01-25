import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QSlider
from PyQt5.QtCore import Qt
from peewee import SqliteDatabase, Model, CharField

# Database setup
db = SqliteDatabase('artprompts.db')

class ArtPrompt(Model):
    content = CharField()

    class Meta:
        database = db

db.connect()
db.create_tables([ArtPrompt], safe=True)

# Define lists for each component of the prompt
adjectives = ["moody", "vibrant", "ethereal", "shadowy"]
nouns = ["portraits", "landscapes", "still life", "abstracts"]
verbs = ["create", "capture", "envision", "paint"]
styles = ["black and white", "sepia-toned", "high-contrast", "HDR"]
camera_features = ["high ISO sensitivity", "long exposure", "wide aperture", "fast shutter speed"]

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Art Prompt Generator')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        # Create sliders for each prompt component
        self.adjective_slider = QSlider(Qt.Horizontal)
        self.noun_slider = QSlider(Qt.Horizontal)
        self.verb_slider = QSlider(Qt.Horizontal)
        self.style_slider = QSlider(Qt.Horizontal)
        self.camera_feature_slider = QSlider(Qt.Horizontal)

        # Set ranges for the sliders from 0 to 100
        self.adjective_slider.setRange(0, 100)
        self.noun_slider.setRange(0, 100)
        self.verb_slider.setRange(0, 100)
        self.style_slider.setRange(0, 100)
        self.camera_feature_slider.setRange(0, 100)

        # Add sliders to the layout
        layout.addWidget(self.adjective_slider)
        layout.addWidget(self.noun_slider)
        layout.addWidget(self.verb_slider)
        layout.addWidget(self.style_slider)
        layout.addWidget(self.camera_feature_slider)

        # Add a generate button
        self.generate_button = QPushButton('Generate Art Prompt')
        self.generate_button.clicked.connect(self.generate_prompt)
        layout.addWidget(self.generate_button)

        # Add a label to display the generated prompt
        self.prompt_label = QLabel("Your generated art prompt will appear here.")
        self.prompt_label.setWordWrap(True)
        layout.addWidget(self.prompt_label)

        # Set the main widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    def generate_prompt(self):
        # Get the current values of the sliders
        adjective_weight = self.adjective_slider.value()
        noun_weight = self.noun_slider.value()
        verb_weight = self.verb_slider.value()
        style_weight = self.style_slider.value()
        camera_feature_weight = self.camera_feature_slider.value()

        # Construct the prompt based on the weights
        prompt = self.construct_weighted_prompt(
            adjectives, nouns, verbs, styles, camera_features,
            adjective_weight, noun_weight, verb_weight, style_weight, camera_feature_weight
        )

        # Update the label text with the generated prompt
        self.prompt_label.setText(prompt)

    def construct_weighted_prompt(self, adjectives, nouns, verbs, styles, camera_features, adj_weight, noun_weight, verb_weight, style_weight, cam_feature_weight):
        # Choose a word from each category based on the weights
        adjective = adjectives[adj_weight % len(adjectives)]
        noun = nouns[noun_weight % len(nouns)]
        verb = verbs[verb_weight % len(verbs)]
        style = styles[style_weight % len(styles)]
        camera_feature = camera_features[cam_feature_weight % len(camera_features)]

        # Construct the prompt
        weighted_prompt = f"{adjective} {noun} {verb} {style} {camera_feature}"
        return weighted_prompt.capitalize()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

