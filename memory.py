if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        print("OPENAI_API_KEY is not set. Please add your key to .env")
        exit(1)
      else:
        print("API key set")

      llm = ChatOpenAI()
      conversation = ConversationChain(
          llm=llm,
          memory=ConversationEntityMemory(llm=llm),
          #memory=ConversationBufferMemory(),
          prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
          verbose = True
      )

      st.write("hello, i am chatgpt cli!")
      
      user_input = response

      ai_response = conversation.predict(input=user_input)

      

      st.write("\nAssistant:\n", ai_response)
      

    
