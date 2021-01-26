import re
import json
from io import StringIO
from io import open
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, process_pdf
from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfinterp import PDFTextExtractionNotAllowed
import os
import jieba

# log
# 2020/12/23 1.04
# 新增label2id方法

# 2020/12/22 1.03
# 完善read_json方法

# 2020/12/7 1.02
# 修改read_pdf_by_box方法 对标题进行多一层判断,如果字符串过长或者有标点符号 则判断成非标题

# 12/7 1.01
# 新增字符串是否包含符号

# 11/20 1.00
# 修改了写入方法

def split_to_paragraph(content, filter_length=(2, 1000)):

    """
        拆分成段落
    """
    content = re.sub(r"\s*", "", content)
    content = re.sub("([。！…？?!；;])", "\\1\1", content)
    sents = content.split("\1")
    sents = [_[: filter_length[1]] for _ in sents]

    res = []
    temp = ''
    for _ in sents:
        if len(temp + _ ) > filter_length[1]:
            res.append(temp)
            temp = _
        else:
            temp = temp + _
    if len(temp) > 0:
        res.append(temp)

    return res

def split_to_sents(content, filter_length=(2, 1000)):

    """
        拆分成句子
    """
    content = re.sub(r"\s*", "", content)
    content = re.sub("([。])", "\\1\1", content)
    sents = content.split("\1")
    sents = [_[: filter_length[1]] for _ in sents]
    return [_ for _ in sents
            if filter_length[0] <= len(_) <= filter_length[1]]

def split_to_subsents(content, filter_length=(2, 1000)):

    """
        拆分成子句
    """

    content = re.sub(r"\s*", "", content)
    content = re.sub("([。！…？?!；;,，])", "\\1\1", content)
    sents = content.split("\1")
    sents = [_[: filter_length[1]] for _ in sents]
    return [_ for _ in sents
            if filter_length[0] <= len(_) <= filter_length[1]]

def write_json(filename,res):
    json_str = json.dumps(res,ensure_ascii=False,indent=4)
    with open(filename, 'w',encoding = 'utf-8') as json_file:
                    json_file.write(json_str) 

def read_json(filename):
    try:
        with open(filename,'r',encoding='utf8') as f:
            json_data = json.load(f)
    except:
        json_data = []
        with open(filename,encoding='utf-8') as f:
            for line in f:
                json_data.append(json.loads(line))        
    return json_data

def read_pdf_by_line(filename):

    """
    read pdf by line

    """
    with open(filename, "rb") as pdf:
            rsrcmgr = PDFResourceManager()
            retstr = StringIO()
            laparams = LAParams()
            device = TextConverter(rsrcmgr, retstr, laparams=laparams)
            process_pdf(rsrcmgr, device, pdf)
            device.close()
            content = retstr.getvalue()
            retstr.close()
            lines = str(content).split("\n")
    return lines

def read_pdf_by_box(pdf):

    praser = PDFParser(open(pdf, 'rb'))
    doc = PDFDocument()
    praser.set_document(doc)
    doc.set_parser(praser)

    doc.initialize()

    if not doc.is_extractable:
        return []
    else:

        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)

        res = []
        nums = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14']

        # marks = ['。','！','…','？','?','!','；',';','，']

        for page in doc.get_pages():

            interpreter.process_page(page)                        
            layout = device.get_result()
            
            temp_label = ''   # 段落标签
            is_merge = False  # 针对pdf的分页

            for x in layout:
                if isinstance(x, LTTextBox):

                    text = x.get_text().strip()

                    if text in nums:
                        is_merge = True
                        continue
                    # print(text)
                    if len(text) < 2:
                        continue
                    
                    if is_merge :
                        is_merge = False
                        if len(res) > 0:
                            res[-1][1] = res[-1][1] + clean_data(text) 
                        temp_label = ''
                        continue

                    if text[0] in nums and text[1] == '.':
                        start = 1
                        for i in range(1,len(text)-1):
                            if text[i] in nums or text[i] == '.':
                                start = i  
                            else:
                                break
                        temp_label = ''
                        text = text[start+1:]   
                    
                    if text == '':
                        continue
                    
                    # 12.7
                    # isHasMark = False
                    # for mk in marks:
                    #     if mk in text:
                    #         isHasMark = True
                    #         break

                    if len(text) < 18 and not isHasMark(text):

                        temp_label = text

                    if '    ' in text:
                        s = text.find('     ')
                        temp_label = text[:s]

                        text = text[s:]

                    if len(text) > 18 or text[-1] == '。':
                        temp_label = clean_data(temp_label)
                        text = clean_data(text)

                        if isHasMark(temp_label) or len(temp_label) > 25:
                            res.append(['', temp_label + text])
                        else:
                            res.append([temp_label,text])
                        temp_label = ''


        texts_only = []
        texts_with_title = []

        isStart = True
        temp = ''
        temp_title = ''

        for k in res :
            if isStart :
                  
                if k[0] != '':
                    if k[1][-1] == '。':
                        temp = temp + k[1]
                        temp = clean_data(temp)
                        temp_title = temp_title + k[0]

                        texts_with_title.append([temp_title,temp])

                        temp_list = split_to_sents(temp,(2,128))
                        for t in temp_list:
                            texts_only.append(t)
                        
                        # reset
                        
                        temp = ''
                        temp_title = ''      
                        isStart = False
                    else:
                        temp = temp + k[1]
                        temp_title = temp_title + k[0]

                if k[0] == '':
                    if k[1][-1] == '。':
                        temp = temp + k[1]
                        temp = clean_data(temp)
                        temp_title = temp_title
                        texts_with_title.append([temp_title,temp])

                        temp_list = split_to_sents(temp,(2,128))
                        for t in temp_list:
                            texts_only.append(t)
                        
                        # reset
                        
                        temp = ''
                        temp_title = ''      
                        isStart = False
                    else:
                        temp = temp + k[1]
                        temp_title = temp_title
            else:
                if k[0] == '':
                    texts_only[-1] = texts_only[-1] + k[1]
                    texts_with_title[-1][1] = texts_with_title[-1][1] + k[1]
                else:
                    temp = k[1]
                    temp_title = k[0]

                    if k[1][-1] == '。':
                        temp = clean_data(temp)
                        texts_with_title.append([temp_title,temp])
                        temp_list = split_to_sents(temp,(2,128))
                        for t in temp_list:
                            texts_only.append(t)
                        temp = ''
                        temp_title = ''       

                    else:
                        isStart = True

        return texts_only,texts_with_title

def clean_data(text):

    temp = ['\uf043','\uf020','\uf076','\uf046','\uf075','\uf06c','\uf09f','\uf0d8','\uf072','\uf077','．','…','„',' ']
    List = re.findall(r'[(]cid.*?[)]',text)
    List = List + temp
    for l in List:
        text = text.replace(l,'')
    text = removeLineFeed(text)
    return text            

def removeLineFeed(text):
    """去除换行 tab键"""
    k = text.replace('\r','').replace('\n','').replace('\t',' ')
    return k

def jieba_cut(text):
    
    """
        jieba分词普通模式
        传入text 返回list
    """
    res = []
    seg_list = jieba.cut(text, cut_all=False)
    for i in seg_list:
        res.append(i)

    return res    

def jieba_add_words(lists):
    for i in lists:
        jieba.suggest_freq(i, True)

def write_text(fileName,lists,model = 'normal'):

    """
        按行写入
    """
    if model == 'normal':
        f = open(fileName,'w',encoding='utf8')
    else:    
        f = open(fileName, 'a',encoding='utf8')
    for i in lists:
        f.write(str(i))
        f.write('\n')
    f.close()    

def reSort(filename,isReverse = True):
    
    """
        对字典或数组重新排序,
        isReverse 默认为true 默认越长的元素排在前面 为False 相反 
    """
    
    a = read_json(filename)
    a.sort(key=lambda a: len(a),reverse = isReverse)
    write_json(filename,a)

def isMask(text):
    # 判断该
    mask_list = ["。","！","…","？","?","!","；",";","，"]
    if text in mask_list:
        return True
    else:
        return False   

def isHasMark(text):
    # 判断文本中是否有符号
    marks = ['。','！','…','？','?','!','；',';','，']

    for i in marks:
        if i in text:
            return True

    return False

def get_file_size(fileName):
    # 获取文件大小
    return os.path.getsize(fileName)

def get_file_list(file_path):
    # 获取文件列表
    return os.listdir(file_path)    

def label2id(labels):
    id2label = dict(enumerate(labels))
    label2id = {j: i for i, j in id2label.items()}
    return id2label,label2id    

