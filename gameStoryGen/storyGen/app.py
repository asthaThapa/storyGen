import gradio as gr
import torch
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2LMHeadModel, TextGenerationPipeline, AutoTokenizer

def story_generator():
    tokenizer = GPT2Tokenizer.from_pretrained('./model_save/')
    model = GPT2LMHeadModel.from_pretrained("./model_save/")

    prompt = "<|startoftext|>"

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    sample_outputs = model.generate(
                                    generated, 
                                    #bos_token_id=random.randint(1,30000),
                                    do_sample=True,   
                                    top_k=50, 
                                    max_length = 300,
                                    top_p=0.95, 
                                    num_return_sequences=1
                                    )
    story = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
    
    return story

def genre_story_gen(storyPlot, genre):
    print(storyPlot)

    input_prompt = "<BOS> <"+genre+">" 

    if(len(storyPlot)>0):
        input_prompt = input_prompt + " " + storyPlot

    model_path = "./story_generator_checkpoint"
  
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)

    story = story_generator(input_prompt, max_length=75, do_sample=True,
               repetition_penalty=1.1, temperature=1.2,
               top_p=0.95, top_k=50)
    
    # Accessing the 'generated_text' value from the list
    generated_text = story[0]['generated_text']

    # Removing '<BOS> <Horror>' from the generated text
    cleaned_text = generated_text.replace("<BOS> <"+genre+">", '').strip()
    return cleaned_text


with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(
        """
        # Generate a game plot with AI!
        Click "Generate Game Synopsis" to get a new game story on every click
        """)
        output = gr.Textbox(label="Output")
        gen_btn = gr.Button("Generate Game Synopsis")
        gen_btn.click(fn=story_generator, outputs=output, api_name="story_generator")


        gr.Markdown(
        """
        # Generate a genre specific story plot with AI!
            Chose a genre and Click "Submit" to get a new story
        """)
        gr.Interface(
            genre_story_gen,
            [
                gr.Textbox(label="Your story plot"),
                gr.Dropdown(
                    ["Fantasy", "Thriller", "Historical novel", "Crime Fiction", "Horror", "Science Fiction"],
                    label="Genre",
                    info="Choose a genre to create your story"
                )
            ],
            ["text"]
        )

if __name__ == "__main__":
    demo.launch()

