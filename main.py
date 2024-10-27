import argparse
import json
import random
import time
from backbone_llms.zhipuai_llm import ZhipuaiChatModel
from backbone_llms.openai_llm import OpenaiModel
from utils.llm_workflow import call_specific_agent

def main(llm_name, experiment_name):
    #glm_model = ZhipuaiChatModel(llm_name=llm_name, experiment_name=experiment_name)
    gpt_model = OpenaiModel(llm_name = args.llm_name, experiment_name = args.experiment_name)

    # 固定的prompt部分
    fixed_prompt = """
    现在你需要根据我给的信息判断学生是否上完课程，其中historystamp的意思是给你的信息截止第多少讲，endstamp是你要预测能不能上到endstamp这一讲，content是学生从第一讲到historystamp的发言，如果为中括号就表示学生这节课没发言或者没上课。现在我会给你这信息你只需要回答我true或者false，
    可能有在一节课上发言了但没有上完的
     {
        "HistoryStamp": 1,
        "EndStamp": 1,
        "ID": "chi-jt22@mails.tsinghua.edu.cn",
        "HistoryContent": [
            "第1讲：['好的老师', '好的', '老师，请问AI的定义是什么']"
        ],
        "GroundTruth": false
    },
    同时，也有可能没发言但是上完了课，比如
        {
        "HistoryStamp": 4,
        "EndStamp": 4,
        "ID": "lid20@mails.tsinghua.edu.cn",
        "HistoryContent": [
            "第1讲：['hd ', '我们开始吧', '肖朝军是谁', '什么是模态', '什么是模态\\n']",
            "第2讲：['什么是神经网络模型']",
            "第3讲：[]",
            "第4讲：[]"
        ],
        "GroundTruth": true
    },一下是一些例子：
    这是比较积极的互动
     {
        "HistoryStamp": 1,
        "EndStamp": 1,
        "ID": "wp20@mails.tsinghua.edu.cn",
        "HistoryContent": [
            "第1讲：['老师好，视频中连续的帧可以转为Token,独立的图像怎么转换为Token?这个怎么理解呢', '好的', '老师，我想问一个与第45页ppt内容有关的问题，大模型学习的第三步是从人类反馈中学习，这是不是也可以看作是一种监督学习呢？人类的反馈，不就是在给数据打标签吗？还有一个问题，就是从人类反馈中学习，需要多大的量才能达到比较好的结果？', '好的', '但是从现在的ai大模型看，反倒是一些创造性的工作比较容易收到挑战，比如Midjourney,sora这些工具的使用，让很多艺术设计岗位的从业者感到了压力', '好的', '好的，老师', '好的']"
        ],
        "GroundTruth": true
    },
    这个虽然话不多，但与课堂相关
    {
        "HistoryStamp": 3,
        "EndStamp": 3,
        "ID": "zheng-l21@mails.tsinghua.edu.cn",
        "HistoryContent": [
            "第1讲：[]",
            "第2讲：['梯度的求解方法', '-0.1是怎么计算得到的，请详细计算']",
            "第3讲：[]"
        ],
        "GroundTruth": true
    },
    还有的时候发言质量不太高但是也是上完课的
     {
        "HistoryStamp": 2,
        "EndStamp": 2,
        "ID": "caoruoqi@xuetangx.com",
        "HistoryContent": [
            "第1讲：['我不懂这一页 ']",
            "第2讲：[]"
        ],
        "GroundTruth": true
    },
        现在请预测 
    """

    # 使用当前时间作为随机种子
    seed_value = int(time.time())
    random.seed(seed_value)
    print("Random Seed:", seed_value)

    # 读取并解析整个JSON文件为一个列表
    with open('test.json', 'r', encoding='utf-8') as file:
        data_list = json.load(file)  # 注意这里使用json.load而不是json.loads

    # 随机生成500个数字
    #random_indices = random.sample(range(len(data_list)), 100)

    correct_predictions = 0
    total_predictions = 0
    true_misclassified = 0  # 新增：true被错误分类的数目
    false_misclassified = 0  # 新增：false被错误分类的数目

    for data in data_list:
        try:
            #data = data_list[index]
            data_without_groundtruth = {key: value for key, value in data.items() if key != 'GroundTruth'}
            variable_prompt = json.dumps(data_without_groundtruth, ensure_ascii=False).rstrip("}") + " }"
            full_prompt = fixed_prompt + variable_prompt

            # 使用合并后的prompt进行查询
            result = call_specific_agent(model=gpt_model, agent_name='general', query=full_prompt)
            print(result)  # 只打印content字段的内容

            if result['content'].lower() == str(data["GroundTruth"]).lower():  # 对比content与GroundTruth，忽略大小写
                correct_predictions += 1
            else:
                # 如果预测错误，根据GroundTruth的真实值增加错误分类计数
                if data["GroundTruth"]:
                    true_misclassified += 1
                else:
                    false_misclassified += 1

            total_predictions += 1
            time.sleep(1)  # 每次循环后等待15秒
        except Exception as e:
            print(f"An error occurred: {e}")
            continue  # 发生错误时跳过当前循环

    # 计算正确率
    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy: {accuracy}%")
    # 打印true和false被错误分类的数目
    print(f"True misclassified: {true_misclassified}")
    print(f"False misclassified: {false_misclassified}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='stb experiment')
    parser.add_argument('--llm_name', type=str, default='glm-4', help='series number of the llm api')
    parser.add_argument('--experiment_name', type=str, default='test', help='name of the experiment, for log only')
    
    args = parser.parse_args()

    main(args.llm_name, args.experiment_name)