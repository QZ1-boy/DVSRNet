import re
import os
import random
f = open('/share22/home/zhuqiang/zhuqiang/MMCNN/data/train/filelist_train_1.txt','r+')  # 
f1 = open('/share22/home/zhuqiang/zhuqiang/MMCNN/data/train/filelist_train_1+.txt','w')  # 
lines = f.readlines()
# print(lines)
def rreplace(self, old, new, *max):
    count = len(self)
    if max and str(max[0]).isdigit():
        count = max[0]
    return new.join(self.rsplit(old, count))

# for line in lines:
# file_data = ""
for num, line in enumerate(lines):
    # lines.append(line)
    if num%32 == 0 :
        print(line)
        f1.write(line)
        # pass
    # elif line[-8:-5]=='000':
    #     f.write(line)
    else:
        rpstr = '%03d' % (num%32)
        print('line[-4:]',line[-4:-1])
        print(rpstr)
        # line[-4:-1] = rpstr
        # f1.write(line.replace(str(line[-4:-1]), rpstr))
        f1.write(rreplace(line, line[-4:-1], rpstr,1))
        # f1.write(line[::-1].replace(line[::-1], rpstr[::-1], 1)[::-1])
        # f1.write(re.sub(line[-4:-1], rpstr,line))
        # f1.write(line)
        # print(line)
        # file_data = + line
        # f.write(file_data)
    

        
    # line = str(line) + '/' + '000'
    # print(line)
    # str_list=list(line)
    # # print(line)
    # # str = "031/030/029/028/027/026/025/024/023/022/021/020/019/018/017/016/015/014/013/012/011/010/009/008/007/006/005/004/003/002/001/000"
    # # newstr = '/%03d'% (num%32)
    # # print(newstr)
    # # line.replace(str, newstr)
    # nPos=str_list.index('\n')
    # # for i in range(32):
    # str_list.insert(nPos,'/000')
    # # str_list.insert(nPos,'/%03d'% i)
    # str_2="".join(str_list)  # 行的替换
    # # str_2.append(str_2)
    # # print(line)

    # f.write(str_2)



# for num, line in enumerate(lines):    # line.append(line)
#     print('num',num)
#     print("line",line)
# list_after = [val for val in lines for i in range(32)]
# for i in range(32):
#     lines.insert(num-i, line) # 在第二行插入
#     # s = ''.join(line)
# for val in list_after:
#     f.write(val)


f.close()



# ; lines[1] = 
# ; num = int(input('请输入你要产生的手机号个数：'))
# ; for i in range(num):
# ;     start = '1861253'
# ;     random_num = str(random.randint(1,9999))
# ;     new_num = random_num .zfill(4) #不够4位就补0,仅对字符串可以使用
# ;     phone_num = start + new_num
# ;     f.write(phone_num+'\n')
# ; f.close()


# ; fp = file('data.txt')

# ; lines = []

# ; for line in fp:

# ; lines.append(line)

# ; fp.close()