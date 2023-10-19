# lawyer_chatbot

The goal of this project is to develop an assistant that will be able to give criminal law qualifications with high accuracy and from different perspectives. Responses will be based on Criminal Code of Ukraine.



Taking into account that the chatbot will process Ukrainian requests and respond in the same language, I chose GPT-3.5 Turbo/4 from openai API as the main model, as it works best with foreign languages. But vanil gpt can't meet requirements. It was trained on 2021 data, so some knowledge is currently outdated, so it is necessary to optimize the model. 
To optimize the LLM model I found 3 main apзroaches: Prompting, Embeddins and, in a pinch, Fine-tuning. In some cases combination of approaches can yield excellent result.
