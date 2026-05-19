“/inspire/qb-ilm2/project/26summer-camp-01/26210893/AutoResearchClaw”下是可以借鉴的开源项目，主要要借鉴一些我需要的成熟的模块。现在我要在当前目录下进行项目优化，主要是“/inspire/qb-ilm2/project/26summer-camp-01/26210893/harness-sii”下的程序。

课题名称：自进化的任务求解智能体
课题亮点：当前大多数 LLM 应用只是单轮或多轮对话，缺乏自主行动与反思能力。真正有价值的智能体应当能够： 自主规划：把一个复杂任务拆解为可执行的子步骤； 调用工具：通过联网搜索、浏览器导航等完成每个步骤； 自我反思与进化：对失败的尝试进行总结，修正策略，再次尝试，而不是无脑重试。

研究/实践目标与预期成果：
1. 智能体搭建：基于 Harness 工程和 ReAct 架构实现一个能完成多步工具调用的最小智能体以及沙盒环境。
2. 无进化基线测试：在 2 个任务（SimpleQA 和 2Wiki）上测试原始智能体，记录成功率，作为基线。
3. 添加反思模块：给智能体添加"任务失败后自动反思"能力，这是进化的核心。
4. 添加记忆模块：在反思的基础上，添加记忆模块，这是进化的动力。
5. 对比评估与分析：重新在 2 个任务上测试智能体，量化"进化"带来的性能提升，评估当前智能体的进化效率。

可能会遇到的挑战有：
1. Agent 容易进入死循环，例如：不断重复搜索，工具调用错误，迟迟得不到最终答案。需要设计合理的 Harness 管理和控制机制。
2. Reflection 和 Memory 不一定有效，并不是所有反思和记忆都能带来提升。需要思考"哪些信息才是真正有价值的反馈？"以及"记忆哪些内容才能真正帮助到之后的任务？" 
3. Benchmark 结果不稳定，不同 Prompt 或模型设置可能影响结果。需要控制变量，重复实验，合理分析
4. 项目时间较紧，建议：优先完成最小可用系统、不追求复杂架构、优先跑通完整闭环

课题介绍：课题评分标准（100 分制）
1. 智能体搭建评分（10 分）
本项目需要搭建一个基于 Harness 工程的 ReAct 智能体，该智能体能够实现多轮工具调用过程：LLM 调输出 tool call，工具执行结果返回，LLM 分析判断是否需要进入下一轮。直到达到最大轮数或者返回最终结果。评分标准是基础智能体在 SimpleVQA 和 2Wiki 两个数据集（各 100 个 case）上的性能表现。注：项目组提供了一套完整的 Harness 框架可供参考，包括 LLM 权重及部署环境以及推理Cases。评分：根据 200 个 case 在原始 Agent 上的性能排名评分
2. 工具搭建评分（5 分）本项目需要搭建以下两种工具：
搜索工具（0-2 分），基于在线搜索工具，根据工具实现的功能种类及效率给分，满分参考标准：能够实现文搜文，图搜文功能。（一个功能 1 分、满分 2 分）浏览器工具（0-3 分），基于沙盒浏览器工具，根据工具实现的功能种类及效率给分，满分参考标准：能够实现访问页面，获取页面文本内容以及并发处理多个页面。（一个功能 1 分、满分 3 分）注：项目组提供了一套完整的浏览器沙箱工具部署，调用函数和搜索 API 工具调用函数。评分：根据工具实现的种类评分
3. 反思模块评分（10 分）
反思模块要求智能体在失败后能够分析失败原因，而非直接重试。评分维度：是否在失败后自动触发 Reflection，是否能定位失败原因，是否能自动生成修正策略，是否能将当前任务上的反思结果运用在之后的任务中等（可搭建为 skill 形式、system prompt、外部模型调用等）注：考虑到一些基模已经拥有一定的反思能力，因此该项指标更考验同学自己新增的反思模块对于先前基模性能的提升作用。评分：根据相关代码的功能完整性，结构多样性和实用性评分
4. 记忆模块评分（10 分）
要求智能体能将历史经验（好经验和坏经验）沉淀为长期知识，指导模型在新任务的推理中学习好的，改正错的。评分维度：是否存在短期或长期记忆，是否具备结构化存储，记忆更新手段，是否能在后续任务中有效调用，是否能减少重复错误等（记忆模块也可搭建为 skill形式等）评分：根据相关代码的功能完整性，结构多样性和实用性评分
5. 进化效率评分 （35 分）
主要从以下五个维度展开，1.准确率提升：有无进化能力模型的准确率对比。2.Token 优化：在 200 个 case 上的 token 数消耗对比。3.推理轮数优化：是否减少失败推理，失败工具调用等无效步骤。4.工具调用优化：是否减少无关工具调用等无效步骤。5.推理时间优化：是否减少了在 200 个 case 上的推理时间。以上列出榜单排名进行评分。此外，还要对 200 个case 的最终性能排名评分：进化机制排名(25 分)，最终结果排名(10 分)
6. 公开打榜评分（20 分）
要求在闭源 Agent Benchmark 上测试。分数计算公式为：max(20 * (1 - (所在队伍排名-1)/参加本项目队伍总数)， 1)，最终分数四舍五入取整数。评分：榜单排名（20 分）
7. 加分题（由评审老师酌情给分）（0-10 分）
你是否由于基座自身的能力不足而感到苦恼，或许是时候试一试基于更大的模型采集到的数据来进行蒸馏了。不允许直接在 200 条 SimpleVQA、2wiki 或者打榜数据上进行蒸馏，一经发现，相关成绩取消。

必选模型（Harness 系统的基座模型）：Qwen3.5-9B
外部模型（可选，主要是协助反思与组织记忆，不允许作为 Harness 基座模型，限制在 32B以下）推荐 Qwen3-32B（/inspire/qb-ilm2/project/26summer-camp-01/26210893/Qwen3-32B）

本项目提供的材料：
3、网页搜索及解析 API
4、浏览器沙箱部署环境
5、来自 SimpleVQA 与 2wiki 的 200 条开源评测集数据。见/inspire/qb-ilm2/project/26summer-camp-01/26210893/harness-sii/eval_simplevqa.py、/inspire/qb-ilm2/project/26summer-camp-01/26210893/harness-sii/run_2wiki.py
6、一套完整的 Harness 智能体系统
7、最终打榜闭源数据集，见/inspire/qb-ilm2/project/26summer-camp-01/26210893/harness-sii/run_benchmark.py

现在给的原始框架的测评simplevqa的命令如下，开两个终端：
conda activate pegp
export CUDA_VISIBLE_DEVICES=0,1
export FLASHINFER_CACHE_DIR=/inspire/qb-ilm2/project/26summer-camp-01/26210893/flashinfer_cache
python -m sglang.launch_server \
--model-path /inspire/qb-ilm2/project/26summer-camp-01/public/Qwen3.5-9B \
--port 8000 \
--tp-size 2 \
--mem-fraction-static 0.8 \
--context-length 262144 \
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder \
--served-model-name Qwen3.5-9B \
--enforce-disable-flashinfer-allreduce-fusion \
--disable-custom-all-reduce

conda activate pegp
python harness-sii/eval_simplevqa.py \
--data-file simpleVQA/SimpleVQA.jsonl \
--data-root simpleVQA \
--llm-url http://127.0.0.1:8000/v1 \
--model Qwen3.5-9B \
--max-steps 20 \
--concurrency 2 \
--result-format minimal \
--overwrite

目前已知的问题：