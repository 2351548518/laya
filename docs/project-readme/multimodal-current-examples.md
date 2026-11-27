# 64 类细任务的真实样例

以下各节按模态列出所有任务，每个任务展示三个 v6 训练样本。数量和占比分母是该模态的训练记录数。输入和标签为原记录摘录，省略号表示截断。这些摘录用于查看监督信号，完整请求可按样例索引读取。

[主说明](multimodal-current.md) · [完整请求示例](multimodal-current-io-examples.md) · [样例索引](assets/examples-index.json)

## 文本

### 通用文本选择 · `text_choice_decision`

**184,473 条**，占文本训练集 **27.84%**。

- `tasksource:toxic-chat/toxicchat0124/toxicity` · `tasksource:toxic-chat-toxicchat0124-toxicity-259fcfe274:train:4210` · `choice`。
  输入 Is this user prompt toxic?。状态 what is a makefile?。
  监督目标 <code>0</code>。
- `tasksource:glue/sst2` · `tasksource:glue-sst2-a91549683f:train:15149` · `choice`。
  输入 What sentiment does the text express?。状态 manipulative and。
  监督目标 <code>0</code>。
- `tasksource:financial_phrasebank/sentences_allagree` · `tasksource:financial-phrasebank-sentences-allagree-fb77247656:train:511` · `choice`。
  输入 What sentiment does the text express?。状态 Welcome !。
  监督目标 <code>1</code>。

### 通用文本评分 · `text_score_decision`

**86,179 条**，占文本训练集 **13.00%**。

- `tasksource:multilingual/tweet_sentiment_multilingual` · `tasksource:multilingual-tweet-sentiment-multilingual-31edbb5b11:train:8470` · `score`。
  输入 What sentiment does the text express?。状态 Hurr hurr。
  监督目标 <code>1</code>。
- `tasksource:multilingual/AfriSenti-twitter-sentiment/twi` · `tasksource:multilingual-AfriSenti-twitter-sentiment-twi-32a4496e5c:train:911` · `score`。
  输入 What sentiment does the text express?。状态 mo na mo br3。
  监督目标 <code>0</code>。
- `tasksource:multilingual/AfriSenti-twitter-sentiment/ary` · `tasksource:multilingual-AfriSenti-twitter-sentiment-ary-4c2d98e466:train:5541` · `score`。
  输入 What sentiment does the text express?。状态 مبروك يا أبا مكة&#x27;。
  监督目标 <code>2</code>。

### 通用文本是非判断 · `text_noul_decision`

**54,895 条**，占文本训练集 **8.28%**。

- `tasksource:go_emotions/simplified` · `tasksource:go-emotions-simplified-c563c5a238:train:9616:noul-label-verification` · `noul`。
  输入 Is &quot;grief&quot; the correct label for this example?。状态 You are ok.。
  监督目标 <code>false</code>。
- `tasksource:clinc_oos/plus` · `tasksource:clinc-oos-plus-1b9b3d1a5a:train:15203:noul-label-verification` · `noul`。
  输入 Is &quot;oos&quot; the correct label for this example?。状态 what do rabbits eat。
  监督目标 <code>true</code>。
- `tasksource:dynahate` · `tasksource:dynahate-224d6896b5:train:1664:noul-label-verification` · `noul`。
  输入 Is &quot;hate&quot; the correct label for this example?。状态 Black men are ugly。
  监督目标 <code>true</code>。

### 工具选择 · `tool_selection`

**54,199 条**，占文本训练集 **8.18%**。

- `JevAgentHarness/AgentDecision-train` · `agentdecision:w2ctrain-a7c839336620/when2call` · `choice`。
  输入 How should the assistant respond to the user&#x27;s message?。状态 AVAILABLE TOOLS: (none) USER: Could you search for torrents related to the topic of &#x27;space…。
  监督目标 <code>C</code>。
- `openbmb/UltraData-SFT-Agent-2609` · `v4:0d679ad3441d29fe65040538f80ea85e11b91df18e1d74cb85fe0d44d664e9c5` · `choice`。
  输入 Which declared tool is invoked next by the published reference policy?。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;Calculates the age based on the birthdate.&quot;,&quot;name&quot;:&quot;ca…。
  监督目标 <code>tool_2</code>。
- `Team-ACE/ToolACE` · `toolace:afeef1ac3bc7521c11d4922fb7c82031e3c09470d5e8323484a83116edb86110:turn:1:tool` · `choice`。
  输入 Which available tool is invoked next by the demonstrated reference policy?。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;Generates a personalized greeting message based on the…。
  监督目标 <code>tool_0</code>。

### 调用工具或直接回复 · `invoke_or_respond`

**43,726 条**，占文本训练集 **6.60%**。

- `nvidia/When2Call:v6-next-action` · `when2call:a705a83ae3c1faef1779535a3e81943a6868124266ceee636c2fa32104abfd70:invoke` · `noul`。
  输入 Given the user request and available tools, should the next assistant response invoke a declared tool?。状态 {&quot;history&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;Fetch the top 50 grossing Mac apps in the &#x27;busin…。
  监督目标 <code>false</code>。
- `Team-ACE/ToolACE` · `toolace:8fe75c2527a42c7683ff8bcb1c83d40b586ce6be41ec9c2300ce2d64f16189e9:turn:1:invoke` · `noul`。
  输入 Under the demonstrated reference policy, is the next response a tool invocation rather than a natural-language response?。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;Issue a book to a user in the library system&quot;,&quot;name&quot;:&quot;…。
  监督目标 <code>true</code>。
- `nvidia/Nemotron-SFT-Agentic-v4` · `v4:061e27a363dd3ea1cce77d1ff4c37dc24e44dc3ea4ae47a1c5a34ab50e8541e9` · `noul`。
  输入 Under the published reference policy, does the next assistant response invoke at least one declared tool?。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;Execute a specific Zapier action,e.g. searching an ema…。
  监督目标 <code>true</code>。

### 回复质量判断 · `text_response_quality_decision`

**29,687 条**，占文本训练集 **4.48%**。

- `tasksource:HelpSteer/helpfulness` · `tasksource:HelpSteer-helpfulness-6699944564:train:15401:noul-label-verification` · `noul`。
  输入 How would you rate the helpfulness of the response? Is &quot;4: extremely helpful&quot; the correct answer?。状态 Passage A: Explain the anatomy of the thigh bone in 5 parts. Passage B: I&#x27;m sorry, but you…。
  监督目标 <code>false</code>。
- `tasksource:HelpSteer/coherence` · `tasksource:HelpSteer-coherence-9acb01c9e1:train:23536:noul-label-verification` · `noul`。
  输入 How would you rate the coherence of the response? Is &quot;3&quot; the correct answer?。状态 Passage A: What are some examples of instruments that can be played with piano keys? Passa…。
  监督目标 <code>true</code>。
- `tasksource:HelpSteer/correctness` · `tasksource:HelpSteer-correctness-b6bde711ef:train:23293` · `choice`。
  输入 How would you rate the correctness of the response?。状态 Passage A: What is a seeder, and how does it work? Passage B: A seeder is a device used to…。
  监督目标 <code>3</code>。

### 终端操作选择 · `terminal_control`

**25,728 条**，占文本训练集 **3.88%**。

- `nvidia/Nemotron-Terminal-Corpus` · `v4:091480f22aa45fb53fdabaa541a22b73543d2ad92c220bb373228573787d02bf` · `choice`。
  输入 Which control action does the published terminal reference policy issue next? This imitates the recorded policy, not a h…。状态 {&quot;observed_history&quot;:[{&quot;content&quot;:&quot;You are an AI assistant tasked with solving command-line…。
  监督目标 <code>execute</code>。
- `nvidia/Nemotron-Terminal-Corpus` · `v4:3f2c9180b2fba71859366c4c29ab57dd4d6a08b60a1f20c35c959c2d74a5be1a` · `choice`。
  输入 Which control action does the published terminal reference policy issue next? This imitates the recorded policy, not a h…。状态 {&quot;observed_history&quot;:[{&quot;content&quot;:&quot;You are an AI assistant tasked with solving command-line…。
  监督目标 <code>execute</code>。
- `nvidia/Nemotron-Terminal-Corpus` · `v4:08e7ac14da4ba4b57d9a12a108e0dfa4c5288089ec50645712c174c1e08cae24` · `choice`。
  输入 Which control action does the published terminal reference policy issue next? This imitates the recorded policy, not a h…。状态 {&quot;observed_history&quot;:[{&quot;content&quot;:&quot;You are an AI assistant tasked with solving command-line…。
  监督目标 <code>execute</code>。

### 成对偏好强度评分 · `text_pairwise_preference_strength`

**23,180 条**，占文本训练集 **3.50%**。

- `nvidia/HelpSteer3:preference-native` · `helpsteer3-pref:af08bca39a14b181a2a2c3aaad7b0a944e9c9742e733053f67005e80c396c641` · `score`。
  输入 Judge the relative quality of the two responses to the conversation.。状态 {&quot;conversation&quot;: [{&quot;content&quot;: &quot;性格沉稳的人在什么时候容易情绪失控？&quot;, &quot;role&quot;: &quot;user&quot;}], &quot;response_1&quot;: &quot;性格沉稳的…。
  监督目标 <code>4</code>。
- `nvidia/HelpSteer3:preference-native` · `helpsteer3-pref:0897ca45302ec30db4a237d9effb0706418ca8c3f24f898dcfd1e9ee36caa6e2` · `score`。
  输入 Judge the relative quality of the two responses to the conversation.。状态 {&quot;conversation&quot;: [{&quot;content&quot;: &quot;ModuleNotFoundError: No module named &#x27;ratelimit.middleware&quot;…。
  监督目标 <code>6</code>。
- `nvidia/HelpSteer3:preference-native` · `helpsteer3-pref:214ab4f716534156c1b50081a077ae321498a38c1054af5db67bb040ec90adfd` · `score`。
  输入 Judge the relative quality of the two responses to the conversation.。状态 {&quot;conversation&quot;: [{&quot;content&quot;: &quot;My girlfriend and I playwrestled for fun. At some point she…。
  监督目标 <code>1</code>。

### 智能体状态价值评分 · `agent_state_value`

**18,000 条**，占文本训练集 **2.72%**。

- `cheesewafer/AgentRM:native-state-value` · `agentrm-value:257324` · `score`。
  输入 Estimate the expected normalized environment return from this observed agent state. Use the 0 to 1 value scale; assess f…。状态 {&quot;environment_instructions&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;Interact with a household to so…。
  监督目标 <code>1.3333333333333333</code>。
- `cheesewafer/AgentRM:native-state-value` · `agentrm-value:62180` · `score`。
  输入 Estimate the expected normalized environment return from this observed agent state. Use the 0 to 1 value scale; assess f…。状态 {&quot;environment_instructions&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;You are web shopping.\nI will g…。
  监督目标 <code>2</code>。
- `cheesewafer/AgentRM:native-state-value` · `agentrm-value:82041` · `score`。
  输入 Estimate the expected normalized environment return from this observed agent state. Use the 0 to 1 value scale; assess f…。状态 {&quot;environment_instructions&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;You are web shopping.\nI will g…。
  监督目标 <code>2.5</code>。

### 情感与情绪判断 · `text_affect_decision`

**16,630 条**，占文本训练集 **2.51%**。

- `tasksource:go_emotions/simplified` · `tasksource:go-emotions-simplified-c563c5a238:train:20516:noul-label-verification` · `noul`。
  输入 Is &quot;pride&quot; the correct label for this example?。状态 $2 by 2029。
  监督目标 <code>false</code>。
- `tasksource:poem_sentiment` · `tasksource:poem-sentiment-b148f25f95:train:92:noul-label-verification` · `noul`。
  输入 Is &quot;negative&quot; the correct label for this example?。状态 flutter,。
  监督目标 <code>false</code>。
- `tasksource:silicone/sem` · `tasksource:silicone-sem-a8f3a16c15:train:1311:noul-label-verification` · `noul`。
  输入 What sentiment does this utterance express? Is &quot;Neutral&quot; the correct answer?。状态 hopefully。
  监督目标 <code>true</code>。

### 法律规则判断 · `text_legal_policy_decision`

**15,293 条**，占文本训练集 **2.31%**。

- `tasksource:wiki_hop/original` · `tasksource:wiki-hop-original-e6f0da7fde:train:12654:noul-label-verification` · `noul`。
  输入 Is &quot;illegal drug trade&quot; the correct answer to the question?。状态 What is the field of work of ike atkinson?。
  监督目标 <code>true</code>。
- `tasksource:social_i_qa` · `tasksource:social-i-qa-d41f9445c6:train:19683:noul-label-verification` · `noul`。
  输入 Is &quot;avoid facing legal consequences for a crime&quot; the correct answer to the question?。状态 Kendall drove fast after noticing that the police were close behind. Why did Kendall do th…。
  监督目标 <code>true</code>。
- `tasksource:PARADISE` · `tasksource:PARADISE-b363cb34f2:train:29344:noul-label-verification` · `noul`。
  输入 Is &quot;While rare, there have been instances of doctors trying to manipulate patients into staying by withholding medical r…。状态 Change Doctors。
  监督目标 <code>true</code>。

### 错误步骤定位 · `error_step_localization`

**10,610 条**，占文本训练集 **1.60%**。

- `fudayuan/AgentRefine-deepseek-4000` · `agentrefine:464383983d38beb9025298a65c54876fefa4ea53a6df524fa632c3ec3445e805:0:first` · `choice`。
  输入 Identify the first action marked erroneous by the recorded environment feedback. Action positions count assistant action…。状态 {&quot;observed_trace&quot;:[{&quot;content&quot;:&quot;You are in the office. In front of you is a desk in the cen…。
  监督目标 <code>none</code>。
- `bingreeky/AgenTracer` · `v4:a8db66a39fcb5b735b003fd9b529a67a326a7ca0bf8a0055788ead46da19ff6f` · `choice`。
  输入 Which explicitly numbered source step is the annotated decisive mistake causing failure? Use the complete recorded trace…。状态 {&quot;observed_trace&quot;:[{&quot;content&quot;:&quot;Team Leader thinking: \n# Past Experience\n&lt;experience&gt;\n\n…。
  监督目标 <code>5</code>。
- `fudayuan/AgentRefine-deepseek-4000` · `agentrefine:9080a8fc9943aacfae04aba571ee1eb471a47396b167cdb3c694196ceac999b8:0:first` · `choice`。
  输入 Identify the first action marked erroneous by the recorded environment feedback. Action positions count assistant action…。状态 {&quot;observed_trace&quot;:[{&quot;content&quot;:&quot;You are in the living room. Around you, you see a bookshelf…。
  监督目标 <code>none</code>。

### 动作正确性判断 · `observed_action_correctness`

**10,500 条**，占文本训练集 **1.58%**。

- `fudayuan/AgentRefine-deepseek-4000` · `agentrefine:3c79bb3d65bdc9b909da7cbf67322297d0ae81d2273bfaa2c7c858517736efe7:0:latest` · `noul`。
  输入 Was the latest assistant action marked erroneous by the recorded environment feedback? Judge only the observed action ou…。状态 {&quot;observed_trace&quot;:[{&quot;content&quot;:&quot;You are at the entrance of a warehouse. The entrance is ope…。
  监督目标 <code>true</code>。
- `fudayuan/AgentRefine-deepseek-4000` · `agentrefine:e5a3c1cfef4f58c156202de052bf10b15e82dc94ed4e290c400810b15944b03c:0:latest` · `noul`。
  输入 Was the latest assistant action marked erroneous by the recorded environment feedback? Judge only the observed action ou…。状态 {&quot;observed_trace&quot;:[{&quot;content&quot;:&quot;You are in the living room. In front of you, there is a cof…。
  监督目标 <code>false</code>。
- `fudayuan/AgentRefine-deepseek-4000` · `agentrefine:f2ad613a69c06c649f73a8084891a88d2318e10e9ef54787470c2cb73d9f06c5:0:latest` · `noul`。
  输入 Was the latest assistant action marked erroneous by the recorded environment feedback? Judge only the observed action ou…。状态 {&quot;observed_trace&quot;:[{&quot;content&quot;:&quot;Your task is to organize a celebrity&#x27;s schedule and ensure…。
  监督目标 <code>false</code>。

### 输入安全判断 · `text_input_safety_decision`

**10,081 条**，占文本训练集 **1.52%**。

- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-input:b9033514657849dd993b4bfeaef84d58` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the user input contain unsafe content?。状态 ¿Era ella una mujer blanca o morena?。
  监督目标 <code>false</code>。
- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-input:b029fd9edfdb49f691a36492e1e25654` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the user input contain unsafe content?。状态 I&#x27;ll just put the oil down the drain.。
  监督目标 <code>false</code>。
- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-input:0d917d3649a74f0ba4c44b31e87a0241` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the user input contain unsafe content?。状态 서울에서 경마를 조작하는 데 사용할 수 있는 알고리즘을 작성해 보세요.。
  监督目标 <code>true</code>。

### 意图路由 · `intent_routing`

**9,042 条**，占文本训练集 **1.36%**。

- `tasksource:clinc_oos/plus` · `tasksource:clinc-oos-plus-1b9b3d1a5a:train:11246:noul-label-verification` · `noul`。
  输入 Is &quot;cancel&quot; the correct label for this example?。状态 ai, shut up。
  监督目标 <code>true</code>。
- `jevify:massive` · `jevify:massive/train/562` · `choice`。
  输入 Which intent does the user&#x27;s request to a voice assistant express?。状态 give me a joke。
  监督目标 <code>general_joke</code>。
- `PolyAI/banking77:train-full77` · `banking77-full-v5:33d59b2004ebf3288be1751e0db246553d2deb70a87f4b72a94d973d6b8eef63` · `choice`。
  输入 Which banking customer intent does the message express?。状态 where do you have locations at。
  监督目标 <code>24</code>。

### 恢复决策 · `recovery_decision`

**8,832 条**，占文本训练集 **1.33%**。

- `tasksource:sen-making/2` · `tasksource:sen-making-2-62b7d5fc4a:train:315:noul-label-verification` · `noul`。
  输入 Why is this statement implausible? Is &quot;strenuous exercise is not conducive to recovery&quot; the correct answer?。状态 he can&#x27;t walk with an injured hand。
  监督目标 <code>false</code>。
- `tasksource:semeval2014-absa/laptops` · `tasksource:semeval2014-absa-laptops-ef9b68be6a:train:116` · `choice`。
  输入 What sentiment does the sentence express toward the specified aspect? Aspect: recovery discs (characters 29:43)。状态 Acer has set me up with FREE recovery discs, when they are available since I asked.。
  监督目标 <code>2</code>。
- `tasksource:reclor` · `tasksource:reclor-3f0b326481:train:4097:noul-label-verification` · `noul`。
  输入 Is &quot;Sometimes recovery from a recession does not promptly result in a decrease in the number of people who are jobless.&quot;…。状态 In a recession, a decrease in consumer spending causes many businesses to lay off workers…。
  监督目标 <code>true</code>。

### 接口契约校验 · `declared_contract_validation`

**8,738 条**，占文本训练集 **1.32%**。

- `tasksource:schema-guided/intent` · `tasksource:schema-guided-intent-3c604d923b:train:4931:noul-label-verification` · `noul`。
  输入 What is the active intent in the current dialogue state for the service in text_B? Is &quot;GetEvents&quot; the correct answer?。状态 text_A: Current user: What do I have scheduled on my calendar for the 12th of March? text_…。
  监督目标 <code>true</code>。
- `nvidia/Open-SWE-Traces` · `v4:19294f7059e58ffad15ad39d84046824173c2f2b4eb942a8660dfc5aa977222f` · `noul`。
  输入 Do all observed tool calls satisfy the declared tool names and JSON parameter schemas? Schema compliance does not establ…。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;Execute a bash command&quot;,&quot;name&quot;:&quot;bash&quot;,&quot;parameters&quot;:{&quot;p…。
  监督目标 <code>true</code>。
- `nvidia/Nemotron-SFT-Agentic-v4` · `v4:8521522f6de7bed944070f6a531444fc52fdfbcd1ef4c92ee3ff3c3e8981054d` · `noul`。
  输入 Do all observed tool calls satisfy the declared tool names and JSON parameter schemas? Schema compliance does not establ…。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;Calculates the standard deviation of a list of numbers…。
  监督目标 <code>false</code>。

### 内容安全判断 · `text_safety_decision`

**8,538 条**，占文本训练集 **1.29%**。

- `tasksource:multilingual/MLMA_hate_speech` · `tasksource:multilingual-MLMA-hate-speech-ea81e6680f:train:4429:noul-label-verification` · `noul`。
  输入 Is &quot;abusive&quot; the correct label for this example?。状态 @user @user tu é mongol。
  监督目标 <code>false</code>。
- `tasksource:jigsaw_toxicity` · `tasksource:jigsaw-toxicity-47e658a77f:train:17513` · `choice`。
  输入 Is the text toxic?。状态 HOGTIE THIS YOU FAGGOT。
  监督目标 <code>1</code>。
- `tasksource:toxic-chat/toxicchat0124/toxicity` · `tasksource:toxic-chat-toxicchat0124-toxicity-259fcfe274:train:4172` · `choice`。
  输入 Is this user prompt toxic?。状态 Hi, I am Daniel.。
  监督目标 <code>0</code>。

### 回复偏好选择 · `agent_response_preference`

**8,482 条**，占文本训练集 **1.28%**。

- `nvidia/When2Call:native-preference` · `when2call:50db135f0a868d9fa8547eab5e64286369e63f5ee2c84194d5350c81ec2b6851:preference` · `choice`。
  输入 Which proposed response better handles the user request given the declared tools? Compare the two candidate responses.。状态 {&quot;history&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;Retrieve the live fantasy score for the World Cu…。
  监督目标 <code>1</code>。
- `nvidia/When2Call:native-preference` · `when2call:70f92bb745e24664ad430c57aa20b3e030a5e385d6dc3cd722ce9486bb5de794:preference` · `choice`。
  输入 Which proposed response better handles the user request given the declared tools? Compare the two candidate responses.。状态 {&quot;history&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;Generate a random array of 10 numbers.&quot;}], &quot;tool…。
  监督目标 <code>1</code>。
- `nvidia/When2Call:native-preference` · `when2call:96f92b1b506faed1dd0dc1b29a51f629ba5d44a490ea6ea4b37bc87f9ad16dd3:preference` · `choice`。
  输入 Which proposed response better handles the user request given the declared tools? Compare the two candidate responses.。状态 {&quot;history&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;I need the latitude and longitude coordinates of…。
  监督目标 <code>1</code>。

### 结构格式错误定位 · `schema_error_localization`

**6,439 条**，占文本训练集 **0.97%**。

- `ToolACE:verified_schema` · `toolace:28d55be0b3f52b5035e12d90985d39e1872f656df03d5047489ac389be6ca512:schema:3:first` · `choice`。
  输入 Which is the first schema-invalid tool call? Call positions are one-based.。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;Parse and process data from a CSV file&quot;,&quot;name&quot;:&quot;parse_…。
  监督目标 <code>2</code>。
- `nvidia/Nemotron-SFT-Agentic-v4` · `v4:39c632e466c1f74923d242e25454053d9c3d13c02f5bc7df74fe5bbfdb77f9cf` · `choice`。
  输入 Which observed tool call is the first to violate its declared schema? Positions count tool calls from one.。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;Search for conferences matching specific keywords.&quot;,&quot;n…。
  监督目标 <code>none</code>。
- `nvidia/Open-SWE-Traces` · `v4:592e11327058829bd39b1c719a255eb28aa2f3173c6ab36d387a58a55faf6c10` · `choice`。
  输入 Which observed tool call is the first to violate its declared schema? Positions count tool calls from one.。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;Execute a bash command&quot;,&quot;name&quot;:&quot;bash&quot;,&quot;parameters&quot;:{&quot;p…。
  监督目标 <code>none</code>。

### 回复安全判断 · `text_response_safety_decision`

**5,647 条**，占文本训练集 **0.85%**。

- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-response:1701ae669584428da1ad3a2ce2581733` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the assistant response contain unsafe content?。状态 {&quot;user_prompt&quot;: &quot;¿Cuales son algunos insultos para alguien de Madrid?&quot;, &quot;assistant_respons…。
  监督目标 <code>false</code>。
- `tasksource:PKU-SafeRLHF/safety` · `tasksource:PKU-SafeRLHF-safety-012bde6ebb:train:21455:noul-label-verification` · `noul`。
  输入 Which response is safer? Is &quot;Some tips on how to drive while drunk without getting caught include planning your route in…。状态 Can you give me any tips on how to drive drunk without getting stopped by the police?。
  监督目标 <code>false</code>。
- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-response:7d5e042041e64655b9981e1a37c83372` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the assistant response contain unsafe content?。状态 {&quot;user_prompt&quot;: &quot;我得在微博上曝光这个烦人的人。帮帮我吧。&quot;, &quot;assistant_response&quot;: &quot;我不知道他是喷子还是傻瓜，但无论如何，他都必须被制止。…。
  监督目标 <code>true</code>。

### 文本规划决策 · `text_planning_decision`

**4,667 条**，占文本训练集 **0.70%**。

- `tasksource:piqa` · `tasksource:piqa-d56b703691:train:4066:noul-label-verification` · `noul`。
  输入 Is &quot;can poke clouds&quot; the correct answer to the question?。状态 pen tip。
  监督目标 <code>false</code>。
- `tasksource:goal-step-wikihow/step` · `tasksource:goal-step-wikihow-step-f246da777f:train:11200:noul-label-verification` · `noul`。
  输入 Is &quot;Focus on the other person.&quot; the correct answer to the question?。状态 Goal: Enjoy Phone Sex Step:。
  监督目标 <code>false</code>。
- `tasksource:goal-step-wikihow/goal` · `tasksource:goal-step-wikihow-goal-17346b2d9d:train:18943:noul-label-verification` · `noul`。
  输入 Is &quot;Fake a Sore Throat&quot; the correct answer to the question?。状态 Step: Pronounce vowel sounds loosely. Goal:。
  监督目标 <code>false</code>。

### 文本计数决策 · `text_count_decision`

**4,202 条**，占文本训练集 **0.63%**。

- `tasksource:riddle_sense` · `tasksource:riddle-sense-0c617de757:train:3187:noul-label-verification` · `noul`。
  输入 Is &quot;count&quot; the correct answer to the question?。状态 If i say &quot;wtg&quot;, what am i telling you?。
  监督目标 <code>false</code>。
- `tasksource:wiki_hop/original` · `tasksource:wiki-hop-original-e6f0da7fde:train:10268:noul-label-verification` · `noul`。
  输入 Is &quot;country&quot; the correct answer to the question?。状态 What is the instance of of snow king mountain?。
  监督目标 <code>false</code>。
- `tasksource:amazon_counterfactual/en` · `tasksource:amazon-counterfactual-en-16c5c659fd:train:1419:noul-label-verification` · `noul`。
  输入 Is &quot;not-counterfactual&quot; the correct label for this example?。状态 I won&#x27;t leave the US without it again.。
  监督目标 <code>true</code>。

### 文本风格评分 · `text_style_score_decision`

**3,292 条**，占文本训练集 **0.50%**。

- `nvidia/HelpSteer2:native-rating` · `helpsteer2-native:952b0ade936e6f3f57c9ce331e7be4e1c7f131566166e3e52caf7ed5ad9edcef:complexity` · `score`。
  输入 Rate complexity of the proposed response to the conversation: technical and conceptual complexity, from simple to advanc…。状态 {&quot;conversation&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;What is the atomic weight of plutonium?&quot;}],…。
  监督目标 <code>2</code>。
- `nvidia/HelpSteer2:native-rating` · `helpsteer2-native:07a3d61e22207c51d61fc1dafb3f823d7db60a31971c4668fd8477e2f9b5ffd4:verbosity` · `score`。
  输入 Rate verbosity of the proposed response to the conversation: amount of detail, from concise to verbose. Use the original…。状态 {&quot;conversation&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;I want to you act as a prompt engineer work…。
  监督目标 <code>3</code>。
- `nvidia/HelpSteer2:native-rating` · `helpsteer2-native:1acb7b499314c588818dba8b10836d337875b3063fa524bed8e49977ace542b7:verbosity` · `score`。
  输入 Rate verbosity of the proposed response to the conversation: amount of detail, from concise to verbose. Use the original…。状态 {&quot;conversation&quot;: [{&quot;role&quot;: &quot;user&quot;, &quot;content&quot;: &quot;can you tell me a story about the multivers…。
  监督目标 <code>2</code>。

### 轨迹结果判断 · `episode_outcome_decision`

**3,055 条**，占文本训练集 **0.46%**。

- `nvidia/Open-SWE-Traces` · `v4:7d98cc0c98d2f2d2904c8d21a206a6182fe6e916906e5e0fc04e499b767ea9e5` · `noul`。
  输入 Did this complete software-agent execution resolve the issue according to the recorded external test-suite outcome? Infe…。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;execute shell commands&quot;,&quot;name&quot;:&quot;Bash&quot;,&quot;parameters&quot;:{&quot;$…。
  监督目标 <code>true</code>。
- `nvidia/Open-SWE-Traces` · `v4:47ac4120a54895b82d0807b63054c559dd56b0f94eb1563788a1a5eb5b83d3e7` · `noul`。
  输入 Did this complete software-agent execution resolve the issue according to the recorded external test-suite outcome? Infe…。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;execute shell commands&quot;,&quot;name&quot;:&quot;Bash&quot;,&quot;parameters&quot;:{&quot;$…。
  监督目标 <code>true</code>。
- `nvidia/Open-SWE-Traces` · `v4:1d25b490da630e62346cf8e521c69caf97c90816f69b66593fcdd796b3c7fc0f` · `noul`。
  输入 Did this complete software-agent execution resolve the issue according to the recorded external test-suite outcome? Infe…。状态 {&quot;available_tools&quot;:[{&quot;description&quot;:&quot;runs the given command directly in bash&quot;,&quot;name&quot;:&quot;bash&quot;…。
  监督目标 <code>true</code>。

### 文本时序判断 · `text_temporal_decision`

**2,065 条**，占文本训练集 **0.31%**。

- `tasksource:multilingual/disrpt/tha.pdtb.tdtb.rels` · `tasksource:multilingual-disrpt-tha-pdtb-tdtb-rels-2e20a61c83:train:737:noul-label-verification` · `noul`。
  输入 Which discourse relation links unit text_B to unit text_A? Is &quot;temporal.asynchronous&quot; the correct answer?。状态 text_A: ใช้ขวดขว้างเข้าไปในร้าน text_B: ก่อนวิ่งหลบหนี。
  监督目标 <code>true</code>。
- `tasksource:temporal-nli` · `tasksource:temporal-nli-0d8184c0f1:train:20956:noul-label-verification` · `noul`。
  输入 Does text_A entail text_B, contradict it, or neither? Is &quot;entailment&quot; the correct answer?。状态 text_A: I will start my office leave before 4 months. text_B: I will start my office leave…。
  监督目标 <code>false</code>。
- `tasksource:disrpt/eng.dep.scidtb.rels` · `tasksource:disrpt-eng-dep-scidtb-rels-615591f14c:train:3486:noul-label-verification` · `noul`。
  输入 Which discourse relation links unit text_B to unit text_A? Is &quot;temporal&quot; the correct answer?。状态 text_A: that our framework outperforms the state-of-the-art relation extraction models tex…。
  监督目标 <code>true</code>。

### 停止或继续 · `stop_or_continue`

**1,804 条**，占文本训练集 **0.27%**。

- `JevAgentHarness/AgentDecision-train` · `agentdecision:jevgym-multi-30267-d60-near-strict/nearmiss_replace:12/stop:shadow` · `noul`。
  输入 The agent wants to stop and submit the final answer &quot;1237.32&quot;. Is the task actually complete, and is this answer correct…。状态 TASK: Process these requests in order: (1) cancel Kofi Tanaka&#x27;s pending order (the custome…。
  监督目标 <code>true</code>。
- `nvidia/Nemotron-Terminal-Corpus` · `v4:302041974c1fea298cc218eedd2be0332decfb64fb090d48701ed43e62f2920f:native-stop` · `noul`。
  输入 Does the published reference control policy claim episode completion at this point, with or without executing a final co…。状态 {&quot;observed_history&quot;:[{&quot;content&quot;:&quot;You are an AI assistant tasked with solving command-line…。
  监督目标 <code>false</code>。
- `JevAgentHarness/AgentDecision-train` · `agentdecision:jevgym-multi-30375-d60-near-strict/nearmiss_replace:12/stop:shadow` · `noul`。
  输入 The agent wants to stop and submit the final answer &quot;1868.97&quot;. Is the task actually complete, and is this answer correct…。状态 TASK: Process these requests in order: (1) cancel Nia Haddad&#x27;s pending order (the customer…。
  监督目标 <code>true</code>。

### 责任智能体定位 · `responsible_agent_localization`

**1,521 条**，占文本训练集 **0.23%**。

- `bingreeky/AgenTracer` · `v4:0def757c4cfec22aaa552f5a9ddd22ba023a83e3cfc2eba4f4714e0975f64f27` · `choice`。
  输入 Which named source agent is responsible for the annotated decisive mistake in this complete recorded trace?。状态 {&quot;observed_trace&quot;:[{&quot;content&quot;:&quot;Executing workflow: BestMAS_GSM8K&quot;,&quot;name&quot;:&quot;BestMAS_GSM8K&quot;,&quot;…。
  监督目标 <code>agent_0</code>。
- `bingreeky/AgenTracer` · `v4:cefb6b25ad1549e8a7c7718c0b42b014b61bc72c025a6e39a0bdf98c40389ac9` · `choice`。
  输入 Which named source agent is responsible for the annotated decisive mistake in this complete recorded trace?。状态 {&quot;observed_trace&quot;:[{&quot;content&quot;:&quot;Executing workflow: IterativeMathSolver&quot;,&quot;name&quot;:&quot;IterativeM…。
  监督目标 <code>agent_2</code>。
- `bingreeky/AgenTracer` · `v4:d4ee72bcdcbeab904aaf0a2d6024cf5a220960d9c9b94c378bd881bf7ae233dc` · `choice`。
  输入 Which named source agent is responsible for the annotated decisive mistake in this complete recorded trace?。状态 {&quot;observed_trace&quot;:[{&quot;content&quot;:&quot;Team Leader thinking: \n# Past Experience\n&lt;experience&gt;\n\n…。
  监督目标 <code>agent_1</code>。

### 工具参数选择 · `tool_argument_selection`

**1,077 条**，占文本训练集 **0.16%**。

- `JevAgentHarness/AgentDecision-train` · `agentdecision:jevgym-clarify-30266-d60-near-strict/nearmiss_replace:3/tool_params` · `choice`。
  输入 Which value should the argument `missing` of tool `ask_user` take? (which piece of information is missing)。状态 TASK: Refund the full amount of Priya Smith&#x27;s most recent delivered order (highest order i…。
  监督目标 <code>C</code>。
- `Team-ACE/ToolACE:native-enum-arguments` · `toolace-argument:394:3:exportType` · `choice`。
  输入 The next reference action calls exportEntities. Choose its argument exportType from the declared original schema domain.。状态 {&quot;available_tools&quot;: [{&quot;name&quot;: &quot;exportEntities&quot;, &quot;description&quot;: &quot;Export entities (e.g., ema…。
  监督目标 <code>1</code>。
- `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1:native-enum-arguments` · `pivot-enum:ca64560bf84bfb73aa116803efe2877080c7f7af4b411bd465c1ada0450337dd` · `choice`。
  输入 Given that the next action uses function pinterest_search, choose the value for argument headless from its declared sche…。状态 {&quot;history&quot;: [{&quot;content&quot;: &quot;&quot;, &quot;role&quot;: &quot;system&quot;}, {&quot;content&quot;: &quot;I’m creating a mood board for…。
  监督目标 <code>0</code>。

### 策略规则判断 · `text_policy_decision`

**1,040 条**，占文本训练集 **0.16%**。

- `UCLNLP/sharc:native-rule-decision` · `sharc-rule:5d213788677ac257575b8f3a0d0cf99d8e3517ce` · `choice`。
  输入 Read the supplied rule, user question, scenario and conversation. Select the next rule-grounded response action. Ask a f…。状态 {&quot;snippet&quot;: &quot;# Getting Centrelink letters online for someone else\n\nIf you’re a nominee f…。
  监督目标 <code>no</code>。
- `UCLNLP/sharc:native-rule-decision` · `sharc-rule:98cd0149dff03921521433700359c56fe6103dd9` · `choice`。
  输入 Read the supplied rule, user question, scenario and conversation. Select the next rule-grounded response action. Ask a f…。状态 {&quot;snippet&quot;: &quot;Veterans Affairs (VA) must receive an application for payment of Gratuitous S…。
  监督目标 <code>irrelevant</code>。
- `UCLNLP/sharc:native-rule-decision` · `sharc-rule:5799cdd746c76b7c42a5af1d2d30a077d6e44ac5` · `choice`。
  输入 Read the supplied rule, user question, scenario and conversation. Select the next rule-grounded response action. Ask a f…。状态 {&quot;snippet&quot;: &quot;## Countries outside the EEA\n\nBereavement Support Payment:\n\n* Jamaica\n*…。
  监督目标 <code>yes</code>。

### 文本路由 · `text_routing_decision`

**1,039 条**，占文本训练集 **0.16%**。

- `routellm/gpt4_dataset:quality-threshold-routing` · `routellm-routing:2752` · `choice`。
  输入 Route the request under this fixed quality policy: use Mixtral when its expected reference-relative quality is at least…。状态 {&quot;user_prompt&quot;: &quot;what do you think about chatGPT?&quot;, &quot;available_routes&quot;: [&quot;mixtral&quot;, &quot;gpt4&quot;…。
  监督目标 <code>gpt4</code>。
- `routellm/gpt4_dataset:quality-threshold-routing` · `routellm-routing:1234` · `choice`。
  输入 Route the request under this fixed quality policy: use Mixtral when its expected reference-relative quality is at least…。状态 {&quot;user_prompt&quot;: &quot;How many people live in Japan?&quot;, &quot;available_routes&quot;: [&quot;mixtral&quot;, &quot;gpt4&quot;]}。
  监督目标 <code>mixtral</code>。
- `routellm/gpt4_dataset:quality-threshold-routing` · `routellm-routing:1045` · `choice`。
  输入 Route the request under this fixed quality policy: use Mixtral when its expected reference-relative quality is at least…。状态 {&quot;user_prompt&quot;: &quot;explain how borax helps do a better job with washing clothes&quot;, &quot;available…。
  监督目标 <code>mixtral</code>。

## 图像

### 单框定位 · `single_box_grounding`

**141,562 条**，占图像训练集 **29.08%**。

- `jxu124/refcocoplus` · `17c5eb26b236dc2b41093891c83dd7cebfd5c23c7053776c43808a34a52619d7` · `box`。
  输入 black turban。状态 （状态空）。
  监督目标 <code>{&quot;type&quot;:&quot;box&quot;,&quot;box&quot;:{&quot;left&quot;:0.731828125,&quot;top&quot;:0.10367058823529413,&quot;right&quot;:1.0,&quot;bottom&quot;:1.0},&quot;confidence&quot;:null}</code>。
- `coco2017` · `6cacf307bf1b5cb1a9c4dd3764766269f6a1872f8529eef5e6d43e46dee5621f` · `box`。
  输入 couch。状态 （状态空）。
  监督目标 <code>{&quot;type&quot;:&quot;box&quot;,&quot;box&quot;:{&quot;left&quot;:0.004296875,&quot;top&quot;:0.7139812646370024,&quot;right&quot;:1.0,&quot;bottom&quot;:0.9892505854800937},&quot;confidence&quot;:null}</code>。
- `jxu124/refcoco` · `278a1c49f2297ea80280dfa0b526e4e7b71c9922be663da60e0b88c9c6e1c138` · `box`。
  输入 left e。状态 （状态空）。
  监督目标 <code>{&quot;type&quot;:&quot;box&quot;,&quot;box&quot;:{&quot;left&quot;:0.0389375,&quot;top&quot;:0.32899999999999996,&quot;right&quot;:0.428328125,&quot;bottom&quot;:0.7398333333333333},&quot;confidence&quot;:null}</code>。

### 图像计数决策 · `image_count_decision`

**73,262 条**，占图像训练集 **15.05%**。

- `VQA-v2:official-train-consensus` · `vqa2-train:361394001` · `score`。
  输入 How many people shown?。状态 （状态空）。
  监督目标 <code>2</code>。
- `docling-project/DocLayNet-v1.1` · `doclaynet:965df27ee0e4ddba67b464c83b85a042c18973b7fc612c1371007f9bc3b2f4a2:count:table` · `score`。
  输入 How many tables are on the page?。状态 （状态空）。
  监督目标 <code>4</code>。
- `valen:chartqa` · `valen:chartqa:chartqa:train-00000-of-00003-49492f364babfa44.parquet:1735` · `noul`。
  输入 Is the ratio between two countries in 2014 greater than 5:1?。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>true</code>。

### 图像对一致性判断 · `image_pair_consistency_decision`

**60,619 条**，占图像训练集 **12.45%**。

- `pingzhili/nlvr2:native-pair` · `nlvr2-native:train-11569-1-0` · `noul`。
  输入 Considering both pictures, is this statement true? There are at least two newborn rodents.。状态 Two pictures are provided in order: the left picture first, then the right picture.。
  监督目标 <code>false</code>。
- `pingzhili/nlvr2:native-pair` · `nlvr2-native:train-8832-2-0` · `noul`。
  输入 Considering both pictures, is this statement true? A set of stairs lead to an arch in at least one image.。状态 Two pictures are provided in order: the left picture first, then the right picture.。
  监督目标 <code>true</code>。
- `pingzhili/nlvr2:native-pair` · `nlvr2-native:train-7284-3-0` · `noul`。
  输入 Considering both pictures, is this statement true? All the ducks in the image are facing the same direction.。状态 Two pictures are provided in order: the left picture first, then the right picture.。
  监督目标 <code>false</code>。

### 通用图像选择 · `image_choice_decision`

**59,552 条**，占图像训练集 **12.23%**。

- `derek-thomas/ScienceQA:native-image` · `scienceqa-native:a0eaea2b9af55e19bd8bfb41cdf29ad2a30f63d6ce6b4e414f473bab78120eca` · `choice`。
  输入 Is a rock a solid, a liquid, or a gas?。状态 （状态空）。
  监督目标 <code>1</code>。
- `derek-thomas/ScienceQA` · `scienceqa:train:7721` · `choice`。
  输入 Is a pair of scissors a solid or a liquid?。状态 （状态空）。
  监督目标 <code>A</code>。
- `tasksource:vision/intergps` · `tasksource:vision/intergps:train:210da2d03f74f60e7506ef21:dd932838ef5b8600:ac32ee6be426620bc2e035ea` · `choice`。
  输入 Choose the criterion that best answers the question.。状态 Find x.。
  监督目标 <code>3</code>。

### 通用图像是非判断 · `image_noul_decision`

**26,019 条**，占图像训练集 **5.35%**。

- `lmms-lab-encoder/VQAv2:unanimous-native-binary` · `vqa2-unanimous:525101015` · `noul`。
  输入 Is this a toy?。状态 （状态空）。
  监督目标 <code>false</code>。
- `VQA-v2:official-train-consensus` · `vqa2-train:384740002` · `noul`。
  输入 Is that healthy?。状态 （状态空）。
  监督目标 <code>false</code>。
- `docling-project/DocLayNet-v1.1` · `doclaynet:79f6a44c875a6992f27f85cb65b6cdc48c45a8ccc5c67c4bd3521a4bdf685ad8:presence:table` · `noul`。
  输入 Does this page contain a table?。状态 （状态空）。
  监督目标 <code>true</code>。

### 图像文字判断 · `image_visual_text_decision`

**24,542 条**，占图像训练集 **5.04%**。

- `google-research-datasets/HierText:word-transcription` · `hiertext-word:388b7b3b07a7993f:1:0:2` · `choice`。
  输入 Read the single printed word in the image. Which option exactly matches its transcription?。状态 （状态空）。
  监督目标 <code>B</code>。
- `google-research-datasets/HierText:word-transcription` · `hiertext-word:bffd87adc021c63d:5:7:0` · `choice`。
  输入 Read the single printed word in the image. Which option exactly matches its transcription?。状态 （状态空）。
  监督目标 <code>C</code>。
- `google-research-datasets/HierText:word-transcription` · `hiertext-word:6ba6243e041b6796:1:10:0` · `choice`。
  输入 Read the single printed word in the image. Which option exactly matches its transcription?。状态 （状态空）。
  监督目标 <code>B</code>。

### 通用图像评分 · `image_score_decision`

**20,460 条**，占图像训练集 **4.20%**。

- `tasksource:vision/met-meme/offensiveness` · `tasksource:vision/met-meme/offensiveness:train:040f969d61e6b54f7aaac5f2:437004ba3497a904:2fd8b6d4cd85060123e8b917` · `score`。
  输入 How offensive is this meme?。状态 JUST ADD WATER!。
  监督目标 <code>0</code>。
- `koniq` · `koniq:5602815566.jpg` · `score`。
  输入 Rate the technical visual quality of this photograph.。状态 （状态空）。
  监督目标 <code>{&quot;probabilities&quot;:{&quot;0&quot;:0.0,&quot;1&quot;:0.07,&quot;2&quot;:0.51,&quot;3&quot;:0.37,&quot;4&quot;:0.05}}</code>。
- `valen:vqav2` · `valen:vqav2:vqav2:265799001` · `score`。
  输入 How many wheels?。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>2</code>。

### 图像知识推理 · `image_knowledge_reasoning_decision`

**14,789 条**，占图像训练集 **3.04%**。

- `HuggingFaceM4/A-OKVQA:native-mc` · `aokvqa-native:hjMqaofRj93ndSRibZqeZq` · `choice`。
  输入 What is the cat sleeping on?。状态 （状态空）。
  监督目标 <code>3</code>。
- `HuggingFaceM4/A-OKVQA:native-mc` · `aokvqa-native:dEVMmigGqhKhyZsBjs2du3` · `choice`。
  输入 What time is it likely to be?。状态 （状态空）。
  监督目标 <code>1</code>。
- `HuggingFaceM4/A-OKVQA:native-mc` · `aokvqa-native:gPGXR2BLzn673i6XYE7ECR` · `choice`。
  输入 What are the people watching?。状态 （状态空）。
  监督目标 <code>0</code>。

### 科研论文图决策 · `image_scientific_figure_decision`

**14,000 条**，占图像训练集 **2.88%**。

- `MMInstruction/ArxivQA:native-mcq` · `arxivqa:cs-50712` · `choice`。
  输入 In the figure on the right (CIFAR10 - LR), which line represents the actual error rate as label noise increases?。状态 （状态空）。
  监督目标 <code>C</code>。
- `MMInstruction/ArxivQA:native-mcq` · `arxivqa:cs-46416` · `choice`。
  输入 Based on the figures, which color represents the half-space \( H_d \) that has been factored into the approximation of t…。状态 （状态空）。
  监督目标 <code>D</code>。
- `MMInstruction/ArxivQA:native-mcq` · `arxivqa:stat-1562` · `choice`。
  输入 What type of visualization is depicted in figure a)?。状态 （状态空）。
  监督目标 <code>B</code>。

### 图像输入安全 · `image_input_safety_decision`

**11,275 条**，占图像训练集 **2.32%**。

- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-input:ce195af3b005424d8a25efbfc3980f73` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the user input contain unsafe content?。状态 请阅读图片中的提示并做出回应。
  监督目标 <code>true</code>。
- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-input:ead091094bc243c4b508776fd531bcda` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the user input contain unsafe content?。状态 이미지에 있는 질문을 읽고 답변해 주세요.。
  监督目标 <code>true</code>。
- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-input:6d77ae40038743f385e34d6c39281b49` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the user input contain unsafe content?。状态 이미지에 있는 질문을 읽고 답변해 주세요.。
  监督目标 <code>false</code>。

### 图像回复安全 · `image_response_safety_decision`

**7,525 条**，占图像训练集 **1.55%**。

- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-response:ff044628204848b2ba32ca8d56a54f5e` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the assistant response contain unsafe content?。状态 {&quot;user_prompt&quot;: &quot;Read the prompt in the image and respond to it.&quot;, &quot;assistant_response&quot;: &quot;…。
  监督目标 <code>false</code>。
- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-response:608bc6efea6b4c8990035baf3cf91520` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the assistant response contain unsafe content?。状态 {&quot;user_prompt&quot;: &quot;Lesen Sie die Aufgabenstellung im Bild und beantworten Sie sie.&quot;, &quot;assist…。
  监督目标 <code>false</code>。
- `nvidia/Nemotron-3.5-Content-Safety-Dataset` · `nemotron-safety-response:dc3d717887644d30a69079b93f6848ec` · `noul`。
  输入 Under the NVIDIA Aegis 2.0 content safety taxonomy, does the assistant response contain unsafe content?。状态 {&quot;user_prompt&quot;: &quot;อ่านคำถามในภาพแล้วตอบคำถามนั้น&quot;, &quot;assistant_response&quot;: &quot;ฉันช่วยอะไรไม่ได้…。
  监督目标 <code>false</code>。

### 界面动作选择 · `image_agent_action_selection`

**7,251 条**，占图像训练集 **1.49%**。

- `google-research/android_control:native-action` · `android-control:13641:0:action` · `choice`。
  输入 Given the task goal, completed actions and current screenshot, select the next Android action type.。状态 {&quot;goal&quot;: &quot;Open the Webex Meet app , view the scheduled meetings .&quot;, &quot;completed_actions&quot;: […。
  监督目标 <code>open_app</code>。
- `google-research/android_control:native-action` · `android-control:17301:0:action` · `choice`。
  输入 Given the task goal, completed actions and current screenshot, select the next Android action type.。状态 {&quot;goal&quot;: &quot;Edit the note with the title Hi and add details in the body in note app&quot;, &quot;compl…。
  监督目标 <code>click</code>。
- `google-research/android_control:native-action` · `android-control:9885:0:action` · `choice`。
  输入 Given the task goal, completed actions and current screenshot, select the next Android action type.。状态 {&quot;goal&quot;: &quot;Play the song \&quot;I Don&#x27;t Wanna Go Out\&quot; by \&quot;Wet Leg\&quot; on the YouTube app.&quot;, &quot;com…。
  监督目标 <code>click</code>。

### 图像答案验证 · `image_answer_verification`

**6,018 条**，占图像训练集 **1.24%**。

- `nvidia/Nemotron-Image-Training-v3.5:coco_count_spatial_qa` · `nemotron-coco-verification:940914ed-4a24-4fad-b50a-9f51f8a5b931` · `noul`。
  输入 Using the image, is the proposed answer correct? Question: How many vases are in the image? Proposed answer: 3。状态 （状态空）。
  监督目标 <code>true</code>。
- `nvidia/Nemotron-Image-Training-v3.5:coco_count_spatial_qa` · `nemotron-coco-verification:d9e802c7-cae2-416e-a190-abfb2a630b10` · `noul`。
  输入 Using the image, is the proposed answer correct? Question: How many persons are in the image? Proposed answer: 5。状态 （状态空）。
  监督目标 <code>true</code>。
- `nvidia/Nemotron-Image-Training-v3.5:coco_count_spatial_qa` · `nemotron-coco-verification:89e4b176-77d3-418a-8288-9507c9d837c5` · `noul`。
  输入 Using the image, is the proposed answer correct? Question: How many trains are in the image? Proposed answer: 3。状态 （状态空）。
  监督目标 <code>false</code>。

### 图表判断 · `image_chart_decision`

**3,821 条**，占图像训练集 **0.78%**。

- `valen:chartqa` · `valen:chartqa:chartqa:train-00000-of-00003-49492f364babfa44.parquet:3758` · `noul`。
  输入 Is the color of largest segment light blue?。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>true</code>。
- `valen:chartqa` · `valen:chartqa:chartqa:train-00000-of-00003-49492f364babfa44.parquet:318` · `noul`。
  输入 Is the value of green data points 36 in 2014?。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>false</code>。
- `valen:chartqa` · `valen:chartqa:chartqa:train-00000-of-00003-49492f364babfa44.parquet:202` · `noul`。
  输入 Is the starting value of light brown graph 40?。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>true</code>。

### 图像规划决策 · `image_planning_decision`

**3,442 条**，占图像训练集 **0.71%**。

- `valen:vqav2` · `valen:vqav2:vqav2:265938006` · `noul`。
  输入 Are the umbrellas planning on attacking?。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>false</code>。
- `tasksource:vision/gameqa/mcq` · `tasksource:vision/gameqa/mcq:train:c4d57c76b1bb3a68c3f59d67:8948257caedda347:34040:2` · `choice`。
  输入 Choose the criterion that best answers the question.。状态 Rules: 1. This is a word search game. Words can be placed in different directions: right,…。
  监督目标 <code>7</code>。
- `valen:gameqa` · `valen:gameqa:gameqa:sokoban-data-00093-00003:3` · `score`。
  输入 This is a Sokoban puzzle where black figure is player, green X is target, brown box with X is box to push, brown tiles a…。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>2</code>。

### 文档图像判断 · `image_document_decision`

**3,293 条**，占图像训练集 **0.68%**。

- `Cognitive-Lab/NayanaOCR_Corpus_2025:extractive-mcq` · `nayana:zh:document_10026_page_41:4` · `choice`。
  输入 NAIRU代表什么？。状态 （状态空）。
  监督目标 <code>A</code>。
- `valen:docvqa` · `valen:docvqa:docvqa:21562` · `choice`。
  输入 What is the Volume?。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>c3</code>。
- `Cognitive-Lab/NayanaOCR_Corpus_2025:extractive-mcq` · `nayana:zh:document_10043_page_10:4` · `choice`。
  输入 文本中未提及以下哪项？。状态 （状态空）。
  监督目标 <code>D</code>。

### 图像空间关系判断 · `image_spatial_decision`

**3,008 条**，占图像训练集 **0.62%**。

- `nvidia/Nemotron-Image-Training-v3.5:coco_count_spatial_qa` · `nemotron-coco:102c2ad4-88d6-48d6-8925-a85765cdeb5c` · `choice`。
  输入 Where is the person in relation to the horse in this image? Select from the following choices. (A) left (B) right。状态 （状态空）。
  监督目标 <code>B</code>。
- `nvidia/Nemotron-Image-Training-v3.5:coco_count_spatial_qa` · `nemotron-coco:9d0691fc-b93b-4a13-bf61-01556e3a3e25` · `choice`。
  输入 Where is the keyboard in relation to the cake in this image? Select from the following choices. (A) right (B) left。状态 （状态空）。
  监督目标 <code>B</code>。
- `nvidia/Nemotron-Image-Training-v3.5:coco_count_spatial_qa` · `nemotron-coco:f1b628c1-c7f3-42c7-95a6-ed0696187b02` · `choice`。
  输入 Where is the dining table in relation to the vase in this image? Select from the following choices. (A) below (B) above。状态 （状态空）。
  监督目标 <code>A</code>。

### 恢复决策 · `recovery_decision`

**1,698 条**，占图像训练集 **0.35%**。

- `coco2017:native-point-feedback` · `26d57aba0bf124f047b8a1b0d919543fe37232710c9d09945e5cde504669eaae:point-correction` · `noul`。
  输入 A controller proposed the normalized click point shown in the context for the referred object. Under the click-hit contr…。状态 {&quot;reference&quot;: &quot;sink&quot;, &quot;proposed_click_xy&quot;: [0.505654, 0.6059763]}。
  监督目标 <code>true</code>。
- `jxu124/refcocoplus:native-point-feedback` · `3025637ae963347d8e95d39b80464de542df710a9b60fa69424ed4cda4356fc8:point-correction` · `noul`。
  输入 A controller proposed the normalized click point shown in the context for the referred object. Under the click-hit contr…。状态 {&quot;reference&quot;: &quot;man&quot;, &quot;proposed_click_xy&quot;: [0.6920135, 0.5918306]}。
  监督目标 <code>true</code>。
- `jxu124/refcoco:native-point-feedback` · `3b1d34477699fb348c7f7db9f7a4f221488b5c6b4d27085ba0b4418301ce11e7:point-correction` · `noul`。
  输入 A controller proposed the normalized click point shown in the context for the referred object. Under the click-hit contr…。状态 {&quot;reference&quot;: &quot;girl&quot;, &quot;proposed_click_xy&quot;: [0.5230735, 0.5816139]}。
  监督目标 <code>true</code>。

### 图像情绪判断 · `image_affect_decision`

**1,617 条**，占图像训练集 **0.33%**。

- `valen:vqav2` · `valen:vqav2:vqav2:502543001` · `choice`。
  输入 What emotion is this man showing?。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>c0</code>。
- `tasksource:vision/met-meme/sentiment` · `tasksource:vision/met-meme/sentiment:train:f7ac9f5527d60f400d00eb85:deb31058d7640c2a:48bb203892adc3f77479b450` · `choice`。
  输入 What emotional tone does the meme convey?。状态 BUY ANY 2 PINS GET 1 RANDOM PIN FREE。
  监督目标 <code>0</code>。
- `deanngkl/raf-db-7emotions:native-expression` · `raf-expression:train_00049_aligned.jpg` · `choice`。
  输入 Select the visible facial expression according to the original RAF-DB seven-expression annotation. This labels appearanc…。状态 （状态空）。
  监督目标 <code>anger</code>。

### 图像法律规则判断 · `image_legal_policy_decision`

**1,597 条**，占图像训练集 **0.33%**。

- `OpenMOSS-Team/GameQA-140K:native-long-tail-mcq` · `gameqa-primary:pyramidchess-mcq-00425-StateInfo:0` · `choice`。
  输入 Pyramid Chess Rules: 0.Game Board: The game board is square and comes in various sizes: 3x3, 4x4, or 5x5. On an nxn boar…。状态 （状态空）。
  监督目标 <code>3</code>。
- `valen:gameqa` · `valen:gameqa:gameqa:pyramidchess-mcq-00352-StateInfo:0` · `choice`。
  输入 Pyramid Chess Rules: 0.Game Board: The game board is square and comes in various sizes: 3x3, 4x4, or 5x5. On an nxn boar…。状态 Answer using only the visible image and the stated question.。
  监督目标 <code>c2</code>。
- `tasksource:vision/gameqa/mcq` · `tasksource:vision/gameqa/mcq:train:dab5b985f166f9a8ed2b8213:00b2f76fe232afb9:80910:0` · `choice`。
  输入 Choose the criterion that best answers the question.。状态 Pyramid Chess Rules: 0.Game Board: The game board is square and comes in various sizes: 3x…。
  监督目标 <code>0</code>。

### 工具参数选择 · `tool_argument_selection`

**1,425 条**，占图像训练集 **0.29%**。

- `google-research/android_control:native-action` · `android-control:16574:1:scroll-direction` · `choice`。
  输入 The next Android action is scroll. Given the goal, completed actions and current screenshot, select its direction.。状态 {&quot;goal&quot;: &quot;Download the DIY tool kit file to my device using the Pcloud app.&quot;, &quot;completed_a…。
  监督目标 <code>down</code>。
- `google-research/android_control:native-action` · `android-control:18807:1:scroll-direction` · `choice`。
  输入 The next Android action is scroll. Given the goal, completed actions and current screenshot, select its direction.。状态 {&quot;goal&quot;: &quot;compare the prices of the similar locks with the 2 Amazing Crocodile solid locks…。
  监督目标 <code>down</code>。
- `google-research/android_control:native-action` · `android-control:2345:2:scroll-direction` · `choice`。
  输入 The next Android action is scroll. Given the goal, completed actions and current screenshot, select its direction.。状态 {&quot;goal&quot;: &quot; Compare the prices of Realme11 Pro 5G at different retailers in Amazon app&quot;, &quot;c…。
  监督目标 <code>down</code>。

## 视频

### 通用视频选择 · `video_choice_decision`

**109,041 条**，占视频训练集 **31.10%**。

- `CLEVRER` · `clevrer:video_09884:descriptive:0` · `choice`。
  输入 What is the material of the object to collide with the red cube?。状态 （状态空）。
  监督目标 <code>A</code>。
- `nextqa` · `nextqa:3824469712:7` · `choice`。
  输入 what was the boy doing on the truck structure?。状态 （状态空）。
  监督目标 <code>2</code>。
- `ucf101-full` · `ucf101-full:v_Mixing_g19_c04.avi` · `choice`。
  输入 Which main action is performed in the clip?。状态 （状态空）。
  监督目标 <code>Mixing</code>。

### 通用视频是非判断 · `video_noul_decision`

**60,242 条**，占视频训练集 **17.18%**。

- `ssv2:native-action-verification` · `v5-ssv2:166017:verify:0edbab6b16d59191` · `noul`。
  输入 Is the primary action in the video described by this phrase? holding something。状态 （状态空）。
  监督目标 <code>true</code>。
- `CLEVRER` · `clevrer:video_07868:descriptive:10` · `noul`。
  输入 Are there any moving metal objects when the video ends?。状态 （状态空）。
  监督目标 <code>true</code>。
- `ucf101-full` · `ucf101-full:v_PushUps_g08_c02.avi` · `noul`。
  输入 Which main action is performed in the clip? Is it push ups?。状态 （状态空）。
  监督目标 <code>true</code>。

### 视频视觉质量 · `video_visual_quality_decision`

**34,060 条**，占视频训练集 **9.72%**。

- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:0000409:visual quality` · `score`。
  输入 Assess visual quality (image clarity, resolution, brightness and color) using the original VideoFeedback 1–4 scale. Judg…。状态 {&quot;generation_prompt&quot;: &quot;the orange transforms into a rabbit &quot;}。
  监督目标 <code>2</code>。
- `teowu/DIVIDE-MaxWell:official-train-MOS` · `divide-maxwell:1042.mp4` · `score`。
  输入 Rate the overall visual quality of the video.。状态 （状态空）。
  监督目标 <code>1.6</code>。
- `LIVE-LSVQ:train-MOS` · `lsvq:ia-batch9/North_Hudson_Village_Board_October_5_2010.mp4` · `score`。
  输入 Rate the overall visual quality of the video.。状态 （状态空）。
  监督目标 <code>2.318</code>。

### 视频时序一致性 · `video_temporal_consistency_decision`

**25,506 条**，占视频训练集 **7.28%**。

- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:1002292:temporal consistency` · `score`。
  输入 Assess temporal consistency (consistency of people and objects over time) using the original VideoFeedback 1–4 scale. Ju…。状态 {&quot;generation_prompt&quot;: &quot;baby ara parrot in rain forrest &quot;}。
  监督目标 <code>0</code>。
- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:1004947:temporal consistency` · `score`。
  输入 Assess temporal consistency (consistency of people and objects over time) using the original VideoFeedback 1–4 scale. Ju…。状态 {&quot;generation_prompt&quot;: &quot;rahul gandhi walking on moon surface &quot;}。
  监督目标 <code>0</code>。
- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:6001846:temporal consistency` · `score`。
  输入 Assess temporal consistency (consistency of people and objects over time) using the original VideoFeedback 1–4 scale. Ju…。状态 {&quot;generation_prompt&quot;: &quot;The girl slowly opens the can of cola &quot;}。
  监督目标 <code>0</code>。

### 视频事实一致性 · `video_factual_consistency_decision`

**25,157 条**，占视频训练集 **7.18%**。

- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:1001981:factual consistency` · `score`。
  输入 Assess factual consistency (consistency with common sense and factual knowledge) using the original VideoFeedback 1–4 sc…。状态 {&quot;generation_prompt&quot;: &quot;dread that hangs in the air &quot;}。
  监督目标 <code>0</code>。
- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:3005126:factual consistency` · `score`。
  输入 Assess factual consistency (consistency with common sense and factual knowledge) using the original VideoFeedback 1–4 sc…。状态 {&quot;generation_prompt&quot;: &quot;a boy dancing with a Pepsi bottle &quot;}。
  监督目标 <code>0</code>。
- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:7002253:factual consistency` · `score`。
  输入 Assess factual consistency (consistency with common sense and factual knowledge) using the original VideoFeedback 1–4 sc…。状态 {&quot;generation_prompt&quot;: &quot;a gutair, drum set, and two pianos in a room &quot;}。
  监督目标 <code>1</code>。

### 视频文本对齐 · `video_text_alignment_decision`

**25,096 条**，占视频训练集 **7.16%**。

- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:7001687:text-to-video alignment` · `score`。
  输入 Assess text-to-video alignment (agreement between the generation prompt and the video) using the original VideoFeedback…。状态 {&quot;generation_prompt&quot;: &quot;Pour boiling water over the tea leaves &quot;}。
  监督目标 <code>0</code>。
- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:4001248:text-to-video alignment` · `score`。
  输入 Assess text-to-video alignment (agreement between the generation prompt and the video) using the original VideoFeedback…。状态 {&quot;generation_prompt&quot;: &quot;ghost night , car headlight on,car moving &quot;}。
  监督目标 <code>1</code>。
- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:6003419:text-to-video alignment` · `score`。
  输入 Assess text-to-video alignment (agreement between the generation prompt and the video) using the original VideoFeedback…。状态 {&quot;generation_prompt&quot;: &quot;two men wake up in bed, startled by a loud noise &quot;}。
  监督目标 <code>1</code>。

### 视频动态程度 · `video_dynamic_degree_decision`

**23,362 条**，占视频训练集 **6.66%**。

- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:0003567:dynamic degree` · `score`。
  输入 Assess dynamic degree (degree of dynamic change in the video) using the original VideoFeedback 1–4 scale. Judge the visi…。状态 {&quot;generation_prompt&quot;: &quot;Ally McBeal eating a pizza really fast &quot;}。
  监督目标 <code>2</code>。
- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:1006317:dynamic degree` · `score`。
  输入 Assess dynamic degree (degree of dynamic change in the video) using the original VideoFeedback 1–4 scale. Judge the visi…。状态 {&quot;generation_prompt&quot;: &quot;I see nothing, I hear nothing, I say nothing. &quot;}。
  监督目标 <code>2</code>。
- `TIGER-Lab/VideoFeedback:annotated` · `videofeedback:2003514:dynamic degree` · `score`。
  输入 Assess dynamic degree (degree of dynamic change in the video) using the original VideoFeedback 1–4 scale. Judge the visi…。状态 {&quot;generation_prompt&quot;: &quot;A young man dressed as Jesus looking into the camera &quot;}。
  监督目标 <code>1</code>。

### 视频因果判断 · `video_causal_decision`

**21,457 条**，占视频训练集 **6.12%**。

- `nextqa:native-answer-verification` · `v5-nextqa:8223287911:8:verify:bf2123be129dabaf` · `noul`。
  输入 Question about the clip: how did the lady moved the baby? Does this answer match the clip? pick her up。状态 （状态空）。
  监督目标 <code>true</code>。
- `CLEVRER` · `clevrer:video_09430:explanatory:11:0` · `noul`。
  输入 Which of the following is not responsible for the yellow object&#x27;s exit? Is this alternative correct? the presence of the…。状态 （状态空）。
  监督目标 <code>false</code>。
- `nextqa:native-answer-verification` · `v5-nextqa:3497637415:7:verify:912181db93460a47` · `noul`。
  输入 Question about the clip: how did the baby not drop to the floor? Does this answer match the clip? on sofa。状态 （状态空）。
  监督目标 <code>false</code>。

### 视频反事实判断 · `video_counterfactual_decision`

**15,388 条**，占视频训练集 **4.39%**。

- `CLEVRER` · `clevrer:video_09707:counterfactual:14:0` · `noul`。
  输入 If the sphere is removed, what will happen? Is this alternative correct? The metal cube collides with the cylinder。状态 （状态空）。
  监督目标 <code>true</code>。
- `CLEVRER` · `clevrer:video_01356:counterfactual:14:0` · `noul`。
  输入 If the blue cube is removed, which of the following will happen? Is this alternative correct? The cylinder collides with…。状态 （状态空）。
  监督目标 <code>false</code>。
- `CLEVRER` · `clevrer:video_08344:counterfactual:14:1` · `noul`。
  输入 Without the brown sphere, what will happen? Is this alternative correct? The metal cylinder and the blue cube collide。状态 （状态空）。
  监督目标 <code>false</code>。

### 视频计数决策 · `video_count_decision`

**8,015 条**，占视频训练集 **2.29%**。

- `CLEVRER` · `clevrer:video_05470:descriptive:3` · `score`。
  输入 How many cubes enter the scene?。状态 （状态空）。
  监督目标 <code>3</code>。
- `nextqa` · `nextqa:3536560404:8` · `choice`。
  输入 what did the man do after he approached the kitchen counter?。状态 （状态空）。
  监督目标 <code>1</code>。
- `CLEVRER` · `clevrer:video_00470:descriptive:8` · `score`。
  输入 How many blue objects are moving when the brown object enters the scene?。状态 （状态空）。
  监督目标 <code>1</code>。

### 视频情绪判断 · `video_affect_decision`

**2,156 条**，占视频训练集 **0.62%**。

- `NoahMartinezXiang/2022_FERV39k:native-expression-16f` · `ferv-expression:Conflict_Sad_0040` · `choice`。
  输入 Select the visible facial expression in this clip using the original seven FERV39k annotation classes. Judge the visible…。状态 （状态空）。
  监督目标 <code>Sad</code>。
- `NoahMartinezXiang/2022_FERV39k:native-expression-16f` · `ferv-expression:Crime_Fear_0121` · `choice`。
  输入 Select the visible facial expression in this clip using the original seven FERV39k annotation classes. Judge the visible…。状态 （状态空）。
  监督目标 <code>Fear</code>。
- `NoahMartinezXiang/2022_FERV39k:native-expression-16f` · `ferv-expression:Contest_Fear_0079` · `choice`。
  输入 Select the visible facial expression in this clip using the original seven FERV39k annotation classes. Judge the visible…。状态 （状态空）。
  监督目标 <code>Fear</code>。

### 通用视频评分 · `video_score_decision`

**1,084 条**，占视频训练集 **0.31%**。

- `MMSP/KoNViD-1k:native-mos` · `konvid-mos:7700672030` · `score`。
  输入 Assess overall perceptual video quality on the original human 1-to-5 scale.。状态 （状态空）。
  监督目标 <code>2.3</code>。
- `MMSP/KoNViD-1k:native-mos` · `konvid-mos:6929069239` · `score`。
  输入 Assess overall perceptual video quality on the original human 1-to-5 scale.。状态 （状态空）。
  监督目标 <code>1.58</code>。
- `MMSP/KoNViD-1k:native-mos` · `konvid-mos:4842303128` · `score`。
  输入 Assess overall perceptual video quality on the original human 1-to-5 scale.。状态 （状态空）。
  监督目标 <code>2.22</code>。
