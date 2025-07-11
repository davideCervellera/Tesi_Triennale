import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
import coremltools as ct
import numpy as np

# Carico il modello e il tokenizer
model_path = "trained_model"
model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)
model.eval()


# Fornisco un esempio di url per fornire un esempio concreto di input
example_text = "http://example.com"
example_input = tokenizer(
    example_text,
    return_tensors="pt",
    max_length=128,
    padding="max_length",
    truncation=True
)

# Wrapper per eliminare problemi con tensori scalar_tensor
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits.float()

wrapped_model = WrappedModel(model)

# Converto il modello in pyThorc
traced_model = torch.jit.trace(
    wrapped_model,
    (example_input["input_ids"], example_input["attention_mask"]),
    strict=False
)
traced_model.save("tinybert.pt")

# Converto il modello in Core ML
dummy_input_ids = example_input["input_ids"].numpy().astype(np.int32)
dummy_attention_mask = example_input["attention_mask"].numpy().astype(np.int32)

mlmodel = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(name="input_ids", shape=dummy_input_ids.shape, dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=dummy_attention_mask.shape, dtype=np.int32),
    ],
    minimum_deployment_target=ct.target.iOS16
)

# Salvo il modello convertito
mlmodel.save("TinyBertURLClassifier.mlpackage")
print("Il modello è stato salvato correttamente")
