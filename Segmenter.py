class Segmenter:
    def __init__(self, dataset_path, model_architecture='unet'):
        self.data_handler = DatasetHandler(dataset_path)
        self.model = SegmentationModel(architecture=model_architecture)

    def train_model(self, epochs=50, batch_size=32):
        trainer = Trainer(self.model, self.data_handler, epochs=epochs, batch_size=batch_size)
        trainer.train()

    def evaluate_model(self):
        evaluator = Evaluator(self.model, self.data_handler)
        evaluator.evaluate()

    def segment_image(self, image):
        # Preprocess image and perform segmentation inference
        preprocessed_image = self.data_handler.preprocess(image, None)
        return self.model.predict(preprocessed_image)

    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
