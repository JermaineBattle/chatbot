import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class Chatbot:
    def __init__(self, model_name='gpt2'):
        """
        Initializes the chatbot with a pre-trained GPT model and tokenizer.
        :param model_name: The name of the model to load from Hugging Face transformers.
        """
        print("Loading the model and tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()  # Set the model in evaluation mode (since we are not training it)

        # Move the model to GPU if available
        if torch.cuda.is_available():
            self.model.cuda()
            self.device = torch.device('cuda')
            print("Model loaded on GPU.")
        else:
            self.device = torch.device('cpu')
            print("Model loaded on CPU.")
    
    def generate_response(self, prompt, max_length=100):
        """
        Generates a response for the given prompt.
        :param prompt: The input text to which the chatbot should respond.
        :param max_length: The maximum number of tokens to generate in the response.
        :return: The generated response as a string.
        """
        # Tokenize the input text
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Generate a response using the model
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)

        # Decode the generated tokens into a string
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return response[len(prompt):].strip()

    def chat(self):
        """
        Starts an interactive chat session with the chatbot.
        """
        print("Chatbot: Hello! I'm an AI-based chatbot. How can I assist you today?")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Chatbot: Goodbye!")
                break
            
            response = self.generate_response(user_input)
            print(f"Chatbot: {response}")

# Create and run the chatbot
if __name__ == "__main__":
    bot = Chatbot(model_name='gpt2')  # Use GPT-2 as the model
    bot.chat()
