import asyncio
import nest_asyncio
from openai import AsyncOpenAI

async def main():

    client = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        # NOTE export NV_API_KEY="nvapi-???" in ~/.bashrc and then 'source ~/.bashrc'
        api_key=os.getenv("NV_API_KEY")
    )
    # https://build.nvidia.com/settings/api-keys NOTE generate the api key here!
    # https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_1-nemotron-70b-reward # NOTE nim list of available models

    n_subtopics = 2
    n_questions = 2
    topic = "機械学習"

    TOPIC_GENERATION_PROMPT_TEMPLATE = """\
    トピックが与えられた場合、そのトピックに関連する {n_subtopics} のサブトピックのリストを生成してください。
    トピックは：{topic}
    リストは番号なしで、サブトピックの説明なしでなければなりません。サブトピックはコンマで区切られる必要があります。リスト以外のテキストは存在してはなりません。
    """

    QUESTION_PROMPT_TEMPLATE = """\
    トピックが与えられた場合、そのトピックに関して{n_questions}個の質問を生成してください。
    トピックは：{sub_topic}
    リスト形式で、質問は改行文字で区切られる必要があります。リスト以外のテキストは存在してはなりません。
    """

    RESPONSE_PROMPT_TEMPLATE = """\
    質問が与えられた場合、その質問に対して考えられる2つの回答を生成してください。
    質問は：{question}
    リスト形式は以下の形式である必要があります：

    RESPONSE A: ここに回答Aのテキストを入力
    RESPONSE B: ここに回答Bのテキストを入力
    """

    # generate sub topics
    async def generate_subtopics(client, topic, n_subtopics):
        prompt = TOPIC_GENERATION_PROMPT_TEMPLATE.format(topic=topic, n_subtopics=n_subtopics)
        response = await client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role" : "user",
                 "content" : prompt}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )
        return response
    subtopics = await generate_subtopics(client, topic, n_subtopics)
    subtopic_list = subtopics.choices[0].message.content.split(",")
    print('1st call of 405b model, output of topic list = ', subtopic_list)
    print("\n\n\n")


    # generate questions of sub topics
    async def generate_questions(client, sub_topic, n_questions):
        prompt = QUESTION_PROMPT_TEMPLATE.format(sub_topic=sub_topic, n_questions=n_questions)
        response = await client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role" : "user",
                 "content" : prompt}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        else:
            print(f"Unexpected response structure: {response}")
            return None

    async def question_generator(client, subtopic_list, n_question):
        tasks = [generate_questions(client, subtopic, n_question) for subtopic in subtopic_list]
        question_list = await asyncio.gather(*tasks)
        return question_list

    nest_asyncio.apply()
    question_list = asyncio.run(question_generator(client, subtopic_list, n_questions))
    print('2nd call of 405b model, output of question list = ', question_list)
    print("\n\n\n")

    # format questions
    question_list_formatted = []
    for question_set in question_list:
        question_list_formatted += question_set.split("\n\n")



    # generate response of each question
    async def generate_responses(client, question):
        prompt = RESPONSE_PROMPT_TEMPLATE.format(question=question)
        response = await client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[
                {"role" : "user",
                 "content" : prompt}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )
        if hasattr(response, 'choices') and response.choices:
            return response.choices[0].message.content
        else:
            print(f"Unexpected response structure: {response}")
            return None

    async def response_generator(client, question_list):
        tasks = [generate_responses(client, question) for question in question_list]
        response_list = await asyncio.gather(*tasks)
        return response_list

    question_response_list = asyncio.run(response_generator(client, question_list_formatted))
    print('3rd call of 405b model, response list = ', question_response_list)
    print("\n\n\n")



    # prepare question:response pair set list
    question_response_pair_list = []
    for question, response_set in zip(question_list_formatted, question_response_list):
        question_response_pair_list.append(
            {
                "question" : question, 
                "responses" : {
                    "response_a" : {"response" : response_set.split("RESPONSE B:")[0].replace("RESPONSE A:", "").strip().split("\n\n")[-1].strip()},
                    "response_b" : {"response" : response_set.split("RESPONSE B:")[-1].split("\n\n")[0].strip()}
                },
            }
        )

    import json

    # export to jsonl file
    with open('synthetic_data.jsonl', 'w') as f:
        for item in question_response_pair_list:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write('\n')


    # running reward scoring model to evaluate the responses
    def get_scores_from_response_orig(openai_response_template):
        print('here:', openai_response_template)
        logprobs = openai_response_template.choices[0].logprobs.content
        score_dict = {}
        for score in logprobs:
            score_dict[score.token] = score.logprob
        return score_dict
    
    def get_scores_from_response(openai_response_template):
        print('here:', openai_response_template)
        #logprobs = openai_response_template.choices[0].logprobs.content
        reward_output = openai_response_template.choices[0].message.content # reward:-14.5625
        key_value = reward_output.split(':')

        score_dict = {}
        #for score in logprobs:
        #    score_dict[score.token] = score.logprob
        if len(key_value) >= 2:
            score_dict[key_value[0]] = float(key_value[1])
        else:
            score_dict['reward'] = -999
        return score_dict

    async def get_response_and_scores(client, question, response_content):
        messages = [
            {"role": "user","content": question},
            {"role": "assistant","content": response_content},]
        response = await client.chat.completions.create(
            #model="nvidia/nemotron-4-340b-reward", # TODO NOTE not exist anymore 2026 03 30
            model="nvidia/llama-3.1-nemotron-70b-reward",
            messages=messages,
        )
        scores = get_scores_from_response(response)
        print('4th call of 70b reward model, scores = ', scores)
        return scores

    # scoring for question:response pair set
    async def process_question_response_pairs(client,question_response_score_list):
        tasks = []
        for question_response_pair in question_response_score_list:
            question = question_response_pair["question"]

            task_a = get_response_and_scores(client, question, question_response_pair["responses"]["response_a"]["response"])
            task_b = get_response_and_scores(client, question, question_response_pair["responses"]["response_b"]["response"])

            tasks.append((task_a, question_response_pair, "response_a"))
            tasks.append((task_b, question_response_pair, "response_b"))
        results = await asyncio.gather(*[task[0] for task in tasks])

        for i, (result, task_info) in enumerate(zip(results, tasks)):
            _, question_response_pair, response_key = task_info
            question_response_pair["responses"][response_key].update(result)
    question_response_score_list = question_response_pair_list.copy()
    await process_question_response_pairs(client, question_response_score_list)
    print(question_response_score_list)

if __name__ == "__main__":
    asyncio.run(main())

