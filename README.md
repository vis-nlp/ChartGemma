# ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild

* Authors: [Ahmed Masry](https://ahmedmasryku.github.io/)*, Megh Thakkar∗ Aayush Bajaj, Aaryaman Kartha, Enamul Hoque, Shafiq Joty (*equal contribution)
* Paper Link: [ChartGemma](https://arxiv.org/abs/2407.04172)


  <img width="724" alt="architecture" src="https://github.com/vis-nlp/ChartGemma/assets/47740795/2aa77ec1-bc5a-445b-8905-e7643ac5996b">


## ChartGemma Model Checkpoints
We release the checkpoint for our pretrained model on huggingface. 
| Task  | Checkpoint Path |
| ------------- | ------------- |
| ChartGemma  | [ChartGemma](https://huggingface.co/ahmed-masry/chartgemma)  |


## Web Demo
If you wish to quickly try our models, you can access our public web demoes hosted on the Hugging Face Spaces platform with a friendly interface!

| Tasks  | Web Demo |
| ------------- | ------------- |
| ChartGemma  | [ChartGemma](https://huggingface.co/spaces/ahmed-masry/ChartGemma) |

## Dataset
You can find our dataset on Huggingface (https://huggingface.co/datasets/ahmed-masry/ChartGemma) 

## Inference
You can easily use our models for inference with the huggingface library! You just need to do the following:

Chage the image_path to your chart example image path on your system

Write the input_text

We recommend using **beam search** with a beam size of 4 to better results, but if your machine's GPU has low memory, you can remove the num_beams from the generate method.


```
from PIL import Image
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import torch

torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

image_path = "/content/chart_example_1.png"
input_text ="program of thought: what is the sum of Faceboob Messnger and Whatsapp values in the 18-29 age group?"

# Load Model
model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Process Inputs
image = Image.open(image_path).convert('RGB')
inputs = processor(text=input_text, images=image, return_tensors="pt")
prompt_length = inputs['input_ids'].shape[1]
inputs = {k: v.to(device) for k, v in inputs.items()}


# Generate
generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)

```

Does you GPU have low memory? The above code is slow on your machine? **We got you covered!** Use the following code that loads the **quantized** version of the model. 
Just make sure to install the following pip modules: bitsandbytes, itsandbytes-cuda112, accelerate

```
from PIL import Image
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image

torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

image_path = "/content/chart_example_1.png"
input_text = "program of thought: what is the sum of Faceboob Messnger and Whatsapp values in the 18-29 age group?"

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

model = PaliGemmaForConditionalGeneration.from_pretrained("ahmed-masry/chartgemma", torch_dtype=torch.float16, quantization_config=bnb_config)
processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

image = Image.open(image_path).convert('RGB')
inputs = processor(text=input_text, images=image, return_tensors="pt")
prompt_length = inputs['input_ids'].shape[1]


# Generate
generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)
```

## Finetuning 
Checkout the example colab notebook in the repo that shows how to finetune the model on the ChartQA Dataset. 
The training code is optimized such that you can train it on a **GPU with 24 GB of memory**. 
The notebook has three different setups LoRA & QLoRA & Full Finetuning. Based on your machine's GPU, you can switch between them. 
This notebook was adapted from [NielsRogge Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb)

# Contact
If you have any questions about this work, please contact **[Ahmed Masry](https://ahmedmasryku.github.io/)** using the following email addresses: **amasry17@ku.edu.tr** or **ahmed.elmasry24653@gmail.com**.

# Reference
Please cite our paper if you use our models in your research. 

```
@misc{masry2024chartgemmavisualinstructiontuningchart,
      title={ChartGemma: Visual Instruction-tuning for Chart Reasoning in the Wild}, 
      author={Ahmed Masry and Megh Thakkar and Aayush Bajaj and Aaryaman Kartha and Enamul Hoque and Shafiq Joty},
      year={2024},
      eprint={2407.04172},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.04172}, 
}
```
