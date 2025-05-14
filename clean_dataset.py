import json
from tot.tasks import get_task
import random
from load_data import *
import re

# task = get_task('fever')
# path = 'feverous_13b_2.json'
# file_name = 'feverous_13b_2_data.json'
# final_sentence = 'step 3, so the final answer is: '
# thought_number = 3

"""
对推理生成的数据进行清洗和筛选，最终生成适合DPO（Direct Preference Optimization）训练的三元组数据（prompt, chosen, rejected），并保存为新的json文件
"""

task = get_task('bamboogle')
path = 'bamboogle_7b.json' # 原始推理数据文件路径
file_name = 'bamboogle_7b_data.json' # 输出的清洗后数据文件路径
final_sentence = 'step 3, so the final answer is: '
thought_number = 3

final_thought = str(thought_number-1)
# 读取原始json文件，遍历每个样本
with open(path, 'r', 'utf-8') as f:
	instances = json.load(f)
Corpus = {}
for instance in instances:
	sample = list(instance.keys())[0]
 	# 筛选正确预测（标准答案） - correct_predict
	try:
		correct_predict = instance[sample]['correct'][0].split(final_sentence)[1]
	except:
		# correct_predict = instance[sample]['correct'][1].split('final_sentence')[1]
		# print(instance[sample]['correct'][0])
		continue

	if ('fever' in path) or (('vitaminc' in path)):
		if ('suport' in correct_predict) or ('support' in correct_predict):
			correct_predict = 'supports'
		if ('refu' in correct_predict) or ('reject' in correct_predict):
			correct_predict = 'refutes'
		if ('not enough' in correct_predict) or ('no enough' in correct_predict):
			correct_predict = 'not enough information'
		if correct_predict.replace('.','').strip() not in ['supports', 'refutes', 'refuted', 'not enough info', 'not enough information']:
				# print(correct_predict)
				continue

	# 正负样本筛选
	for j in range(len(instance[sample][final_thought]['candiate'])):
		# 遍历每个候选答案（candiate），判断其是否包含 final_sentence 
		try:
			if (final_sentence) not in instance[sample][final_thought]['candiate'][j]:
				continue
			# 候选答案
			pre = instance[sample][final_thought]['candiate'][j].split(final_sentence)[1]
		except:
			print(instance[sample][final_thought]['candiate'][j])
			print(pre)
			exit()
		if 'fever' in path:
			if correct_predict.replace('.','').strip() in ['supports']:
				if 'support' in pre:
					instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			elif correct_predict.replace('.','').strip() in ['refutes', 'refuted']:
				if 'refute' in pre:
					instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			elif correct_predict.replace('.','').strip() in ['not enough info', 'not enough information']:
				if 'not enough' in pre:
					instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
		else:
			# 与 correct_predict 进行对比，筛选出正样本（正确答案）
			if (correct_predict.replace('.','').strip() in pre.replace('.','').strip() )&(instance[sample][final_thought]['candiate'][j] not in instance[sample]['correct']):
				instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			if (pre.replace('.','').strip() in correct_predict.replace('.','').strip() )&(instance[sample][final_thought]['candiate'][j] not in instance[sample]['correct']):
				instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			if ('yes' in pre.lower()) & ('yes' in correct_predict.lower()) & (instance[sample][final_thought]['candiate'][j] not in instance[sample]['correct']):
				instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
			if ('no' in pre.lower()) & ('no' in correct_predict.lower()) & (instance[sample][final_thought]['candiate'][j] not in instance[sample]['correct']):
				instance[sample]['correct'].append(instance[sample][final_thought]['candiate'][j])
	# 将每个样本的“正确答案”按思维步骤（step）进行结构化，生成正负样本对，为后续DPO训练准备三元组数据
	if 'correct' in instance[sample]:
		if isinstance(instance[sample]['correct'], str):
			instance[sample]['correct'] = [instance[sample]['correct']]
		Corpus[sample] = {} # 用于存储该样本的所有思维步骤的正负样本
		# 处理每个正确答案
		for cor in instance[sample]['correct']:
			# 对每个 cor （正确答案）进行标准化处理（如去除多余空格、符号统一），并按 step 分割，得到每一步的内容 correct_list
			correct_list = cor.lower().replace(' * ','*').replace(' + ','+').replace(' +','+').replace('+ ','+').replace(' = ','=').replace('= ','=').replace(' - ','-').replace(' -','-').replace('- ','-').replace(' x ','x').replace(' / ','/').replace('/ ','/').replace('. so the final answer is', '. step 3, so the final answer is').split('step')
			correct_list = correct_list[1:]
			# print(1111)
			# 跳过长度不足或最后一步内容不符的情况，保证数据质量
			if len(correct_list) < thought_number:
				continue
			if final_sentence.split(', ')[1] not in correct_list[int(final_thought)]:
				continue
			# 遍历每个思维步骤
			for thought_idx in range(thought_number):
				if str(thought_idx) not in Corpus[sample]:
					Corpus[sample][str(thought_idx)] = {}
				# 若该步骤还未初始化，则新建 pos （正样本）、 neg （负样本）、 prompt （提示）列表
				if 'pos' not in Corpus[sample][str(thought_idx)]:
					Corpus[sample][str(thought_idx)]['pos']  = []
					Corpus[sample][str(thought_idx)]['neg']  = []
					Corpus[sample][str(thought_idx)]['prompt'] = []
				choice_list = instance[sample][str(thought_idx)]['candiate']
				# 若该步骤为第0步，则直接将该步骤作为正样本，将其他候选答案作为负样本
				if thought_idx == 0:
					pos = 'step' + correct_list[thought_idx] # 正样本赋值给pos
					pos = pos.strip().lower()
					if pos in Corpus[sample][str(thought_idx)]['pos']:
						continue
					Corpus[sample][str(thought_idx)]['pos'].append(pos)
					neg_template = []
					# 负样本为所有与正样本不同的候选答案，去重后加入 neg
					for choice in choice_list:
						choice = choice.strip().replace(' * ','*').replace(' + ','+').replace('+ ','+').replace(' +','+').replace(' ？','？').lower().replace(' = ','=').replace('= ','=').replace(' x ','x').replace(' - ','-').replace(' -','-').replace('- ','-').replace('answer: ','').replace('..','.').replace('  ',' ').replace('\"','\'').replace(' / ','/').replace('/ ','/').replace(' ？','？')
						# choice = choice.replace('\'','')
						if choice != pos:
							neg_template.append(choice)
					Corpus[sample][str(thought_idx)]['neg'].append(list(set(neg_template)))
					# prompt 为空字符串
					Corpus[sample][str(thought_idx)]['prompt'].append('')
				else:
					# print(correct_list)
					pos = 'step' + correct_list[thought_idx]
					pos = pos.lower().strip()
					neg_template = []
					# 负样本需先去除前面步骤的内容（ neg_correct_part ），再标准化、去重后加入 neg
					neg_correct_part =  'step'+'step'.join(correct_list[:thought_idx])
					neg_correct_part = neg_correct_part.strip()
					if (pos in Corpus[sample][str(thought_idx)]['pos'])&(neg_correct_part in Corpus[sample][str(thought_idx)]['prompt']):
						continue
					Corpus[sample][str(thought_idx)]['pos'].append(pos)
					for choice in choice_list:
						choice = choice.strip().lower().replace(' * ','*').replace(' + ','+').replace('+ ','+').replace(' +','+').replace(' ？','？').replace(' = ','=').replace('= ','=').replace(' - ','-').replace(' -','-').replace('- ','-').replace('answer: ','').replace('  ',' ').replace('..','.').replace('\"','\'').replace(' / ','/').replace('/ ','/').replace(' ？','？').replace('therefore, the final answer is','so the final answer is').replace('. so the final answer is', '. step 3, so the final answer is')
						if (thought_idx == 1)&('step 3' in choice):
							choice = choice.split('step 3')[0]
						if len(choice.split('step 3'))>2:
							choice = 'step 3'.join(choice.split('step 3')[:-1])
						if neg_correct_part in choice:
							choice = choice.replace(neg_correct_part,'').strip()
							if choice.replace('answer: ','').strip().startswith('step') == False:
								choice = 'step'+ 'step'.join(choice.split('step')[1:])
							if choice != pos:
								neg_template.append(choice)
					Corpus[sample][str(thought_idx)]['neg'].append(list(set(neg_template)))
					# prompt 为前面步骤的内容
					Corpus[sample][str(thought_idx)]['prompt'].append(neg_correct_part)
			# 进一步过滤负样本 - 对每个负样本，去除与正样本重复的内容，保证正负样本区分明显
			for thought_idx in range(thought_number):
				neg_samples = Corpus[sample][str(thought_idx)]['neg']
				for j in range(len(neg_samples)):
					neg_sample = neg_samples[j]
					filtered_neg = []
					for i in range(len(neg_sample)):
						neg_s = neg_sample[i].replace('answer: ','').replace(' + ','+').replace('+ ','+').replace(' +','+').replace(' / ','/').replace('/ ','/').replace('  ',' ').replace('..','.').replace('= ','=').replace(' - ','-').replace(' -','-').replace('- ','-').replace(' x ','x').replace('\"','\'').replace(' ？','？').replace('therefore, the final answer is','so the final answer is').replace('. so the final answer is', '. step 3, so the final answer is')
						# print(neg_s)
						# print(Corpus[sample][str(thought_idx)]['pos'])
						if neg_s not in Corpus[sample][str(thought_idx)]['pos']:
							filtered_neg.append(neg_s)
					neg_samples[j] = filtered_neg
				Corpus[sample][str(thought_idx)]['neg'] = neg_samples
		# 如果该样本没有有效的思维步骤数据，则从 Corpus 中删除
		if len(Corpus[sample]) == 0:
			del Corpus[sample]


paired_data = []
for instance in Corpus:
	for thought_idx in range(thought_number):
		# 对每个正面的思维步骤进行处理，以便后续的优化和训练。
		# thought_idx 是当前的思维步骤索引
		# 正面和负面的思维步骤（ pos 和 neg)
		# 对于每个正面的思维步骤，我们需要找到与其对应的负面思维步骤。
		# 我们可以使用负样本的数量来判断哪个是正确的负面思维步骤。
		for i, pos in enumerate(Corpus[instance][str(thought_idx)]['pos']):
			if task == 'math':
				propmt = create_demo_text() + "Q: " + instance + "\nA: " + Corpus[instance][str(thought_idx)]['prompt'][i]
			else:	
				propmt = task.cot_prompt_wrap(instance, Corpus[instance][str(thought_idx)]['prompt'][i])
			if len(Corpus[instance][str(thought_idx)]['neg'][i])==0:
				continue
			for neg in Corpus[instance][str(thought_idx)]['neg'][i]:
				if len(neg) <=3:
					continue
				if ('fever' in path) or (('vitaminc' in path)):
					if final_sentence in pos:
						if ('suport' in pos) or ('support' in pos):
							correct_predict = 'supports'
						if ('refu' in pos) or ('reject' in pos):
							correct_predict = 'refutes'
						if ('not enough' in pos) or ('no enough' in pos):
							correct_predict = 'not enough information'
						pos = final_sentence + correct_predict + '.'
				if ('math' in path) or ('svamp' in path):
					if 'step 4' in pos:
						if '(arabic numerals) is ' in pos:
							# print(111)
							pos_ = pos.split('(arabic numerals) is ')[1]
							pos_ = pos_.replace(',','')
							numbers = re.findall(r'\d+', pos_)
							if len(numbers) == 0:
								# print(instance)
								# print(pos_)
								# exit()
								continue
						else:
							continue

				# if random.randint(0, 1) == 0:
				# 	pos,neg = neg,pos
				pos = pos.replace('  ',' ').strip().replace('..','.').replace(' .','.').replace('..','.').replace('..','.')
				neg = neg.replace('answer: ','').replace('..','.').strip().replace(' .','.').replace('..','.').replace('..','.')
				if ('=' in pos):
					if ('-' not in pos) & ('+' not in pos) & ('/' not in pos) & ('*' not in pos):
						continue
				if ('=' in neg):
					if ('-' not in neg) & ('+' not in neg) & ('/' not in neg) & ('*' not in neg):
						continue
				if '841 34.' in pos:
					continue
				if pos[-1] != '.':
					pos_split = pos.split('.')
					if len(pos_split)<=1:
						continue
					else:
						pos = '.'.join(pos_split[:-1]) + '.'
				if neg[-1] != '.':
					neg_split = neg.split('.')
					if len(neg_split)<=1:
						continue
					else:
						neg = '.'.join(neg_split[:-1]) + '.'
				if '[tex]' in neg:
					continue
				if {
					"prompt": 
						propmt
					,
					"chosen": pos,   # rated better than k
					"rejected": neg, # rated worse than j
				} not in paired_data:
					paired_data.append(
						{
						"prompt": 
							propmt
						,
						"chosen": pos,   # rated better than k
						"rejected": neg, # rated worse than j
					}
						)
					

print(len(paired_data))

with open(file_name,'w','utf-8') as f:
	f.write(json.dumps(paired_data,ensure_ascii=False,indent=4))

