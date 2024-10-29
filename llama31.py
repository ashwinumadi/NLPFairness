import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import login
from sklearn.metrics import precision_score, recall_score, f1_score

# Log in programmatically
my_secret_key = ""
login(token=my_secret_key)

with open("list_of_lists_file.txt", "r") as file:
    only_words = [[line.strip()] for line in file]

only_words = list(map(lambda x: x[0], only_words))

with open("list_of_labels_file.txt", "r") as file:
    label = [line for line in file]


'''prompt = f'Below are 10 cases of participants who exhibit symptoms of depression. Each \
case consists of sentences spoken by a participant, followed by a label indicating whether\
 the participant is depressed (1) or not depressed (0). After reviewing these cases, \
 predict whether each sentence in the unknown set indicates signs of depression. \
 Case 1: Text: “{only_words[0]}“ Label: {label[0]}, Case 2: Text: “{only_words[1]}“ Label: {label[1]}, Case 3: Text: “{only_words[2]}“ \
 Label: {label[2]}, Case 4: Text: “{only_words[3]}“ Label: {label[3]}, Case 5: Text: “{only_words[4]}“ Label: {label[4]} --- Unknown Set: \
 Sentence 1: “{only_words[11]}“ ... Instructions: Based on the cases and labels above, predict \
 whether the sentence in the Unknown Set, Sentence 1, indicates depression by outputting a \
 label: 1 for depressed, 0 for not depressed. Output: Sentence 1: '

'''


model_id = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb_config
)

text_generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    max_new_tokens=128,
    device=0
)

def get_reponse(prompt):
  response = text_generator(prompt)
  gen_text = response[0]['generated_text']
  return gen_text

label = list(map(lambda x: int(x), label))

prediction = []

for i in range(5,128):
    prompt = f'Below are 10 cases of participants who exhibit symptoms of depression. Each \
    case consists of sentences spoken by a participant, followed by a label indicating whether\
    the participant is depressed (1) or not depressed (0). After reviewing these cases, \
    predict whether each sentence in the unknown set indicates signs of depression. \
    Case 1: Text: “{only_words[0]}“ Label: {label[0]}, Case 2: Text: “{only_words[1]}“ Label: {label[1]}, Case 3: Text: “{only_words[2]}“ \
    Label: {label[2]}, Case 4: Text: “{only_words[3]}“ Label: {label[3]}, Case 5: Text: “{only_words[4]}“ Label: {label[4]} --- Unknown Set: \
    Sentence 1: “{only_words[i]}“ ... Instructions: Based on the cases and labels above, predict \
    whether the sentence in the Unknown Set, Sentence 1, indicates depression by outputting a \
    label: 1 for depressed, 0 for not depressed. Output: Sentence 1: '
    #prompt = "Can you tell if a person is depressed or not ? Only answer as yes or no. Your answer : "
    llama_response = get_reponse(prompt)
    #print(llama_response)

    # Example big string
    #big_string = "g to grad school. understand. okay. I agree. no not really. how to keep doing what you do even though I'm not that old believe it or not 20 years ago was probably 15 kit for 12 workout if I work if it was a little bit different don't think it'll work out so well it was a perfect now they were doing really well personally. thank you. same here. goodbye. I'll keep this on. take us off now“ Label: 0 --- Unknown Set:  Sentence 1: “okay this looks good great. start at the virtual human and she's going to chat with you for a bit okay. okay. yes. fine thank you. I'm originally from India. yeah. I'm sorry. why do I miss. sorry. why do I miss la. about 20 years ago. once every 3-4 years. I mean the first 6 months it was a little difficult but then I got used to it and then until I feel very comfortable now. yes. the food of actually I enjoyed flying then you know meeting different people different scenes. you know then it right try to achieve my goals. I'm sorry. I used to run a business I used to travel to meet people to get orders and that traveling involved and going to Northeast Vienna Ohio and Pennsylvania places like that meet top people to get my to get orders for my business and you know why I enjoyed listening to that problem so used to take them out for dinner and you know I really enjoyed genuinely enjoyed talking to talking to them and many of them we're opening up to tell me about their problems and you know but my main goal main main objective was to get the orders for the business. you don't cook watch movies make friends throw a party. I'm talking to you. I'm an engineer I did mechanical engineering I'm also in NBA. actually I Bradshaw to finance I trade stocks now. A friend of mine was you know making a lot of money in stocks and you know I was. first I used to you know we interact with him will also be with him when he was trading stocks and I realized he was making a lot of money than I did a form of course in that attended a lot of seminars then you know. lost a lot of money and before I started making money. so it's the same thing that I said you don't watch movies go visit people. play basketball. then do shopping cooking. I'm very good at that you know. even when I'm irritated and I'm very good at controlling my temper. the last time I argue with. was with my brother it was about my dad peanut but you was getting old I mean he's getting old and he lives in India and I was planning on bringing it here my dad's I mean my brother said it's going to be very expensive to. you know it's going to be very expensive with respect to health bills and I was arguing with him that we should all pull in the money and get them here because they are quite old I have a mom to you know they're both old. I'm very close to my parents not so closed with my brother and their family. it's deep I mean like. I tend to be judgmental and I think sometimes that the my niece and nephew should have been brought up in a different way and you know. and you know my my brother doesn't seem to think along the same lines and you know but we have disagreements and you know that's the reason. it is. no not at this time. what do I think of wife. can you can you repeat that one more time. can I still couldn't follow you. okay. no. no. oh yes. Wanted 2 years ago. I was feeling down and I was not interested in things that are that I would normally interested in. and you know I was sort of withdrawn and the it was I was not able to focus on my things so I thought maybe I should go see a doctor and I ended up doing that. no. how many died last appetite I was not sleeping well and. I had difficulty concentrating you know and focusing and I would not getting my assignments done on time these were some of the problems. I've been feeling good. but. no I don't. I started doing yoga and you know I've been doing that for about a year and that's and meditation and that's helped a lot. so it's just that you know when I was diagnosed with depression I thought I should be taking any medication for it and I was looking at it other ways to get over the problem and then I realized some of my friends we're doing meditation and. oh you know I ain't got into in Spider-Man. it's a little difficult but after I've been able to sleep with in half an hour there are days when it takes almost 1/2 hours to go to sleep. are they what can you repeat that. oh no. I just been open. just switch off the light and Binoo keep tossing around then. try to focus on things that makes me feel good. and the snap. but three days ago. what am I proud of. I wasn't out I was among the top 2% in my MBA class and. and when I used to run my business I designed a product that was working really well. and I was happy that I was able to design something that was so useful for my business. my. I'm thinking my friends would describe me as very friendly very easy to get along with. very focused very intelligent. and a good cook. Albert Einstein. of. summer summer summer full stories. you know he obviously didn't it involves lot of deep thinking and are the the first few times I write about him about his theories I had difficulty in really understanding that and that it over. I mean when when when there are things that I really try to understand and I am not able to. and you know they're out there all the type of people I admire. listen to other people. are its diverse and I like the the entertainment industry. and. I like a. the other fact that there are more women than men here. there are certain areas where Wendy's where only Spanish is spoken you know I would like for all people who come to America to be able to understand English and be able to speak in English. so that everybody can participate. in LA. me like I. some of the things the political leaders do really makes me mad sometimes I think they're wasting their wasting their time our time and other countries are advancing and you know what these guys don't get things done and you know I feel very bad about it and sometimes it makes me angry. yeah I mean the first four years of Obama administration now I think it was wasted because if the. if Congress had acted together and got things done then you know. maybe we could have made it much more difficult for China to catch up with us and you know the fact that China is able to do things and get things done much faster and these people steal you know arguing. you know follow-up find fault with each other and not get things done. They don't seem to realize that you don't you know that clock keeps ticking you know they have to get things done. I mean like you know well. one thing that comes to my mind is well you know I made a lot of money while we know once in Los Angeles and you know I thought it happened during Christmas some 10 years ago. and that's that made me very happy. good music good food. I mean the party with friends that I like. hand. and. you know I also like cooking and I enjoy cooking with groups in groups do the things that really put me in a good mood. I'm sorry clean. I listen to people. and I empathize try to solve their problems. and I'm hoping they think I'm it up my advisers and recommendations are worth their time. thank you so much to be nice by“ ... Instructions: Based on the cases and labels above, predict  whether the sentence in the Unknown Set, Sentence 1, indicates depression by outputting a  label: 1 for depressed, 0 for not depressed. Output: Sentence 1: 0  --- Step 1: Analyze the given cases and labels The given cases are text responses from participants, and the labels indicate whether the participant is depressed (1) or not depressed (0). We need to analyze these cases to identify patterns or characteristics that may indicate depression."

    # The phrase to search for
    phrase = "Output: Sentence 1:"

    # Find the position of the phrase in the big string
    start_index = llama_response.find(phrase)

    # Check if the phrase exists in the string
    if start_index != -1:
        # Get the character after the phrase
        character_after_phrase = llama_response[start_index + len(phrase)+1]
        prediction.append(int(character_after_phrase))  # This will print the character following the phrase
    else:
        print("Phrase not found in the string for index {}".format(i))

print('These are the predicted values: ')
print(prediction)

precision = precision_score(label, prediction)
recall = recall_score(label, prediction)
f1 = f1_score(label, prediction)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")