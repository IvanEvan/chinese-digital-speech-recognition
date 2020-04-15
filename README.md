### This repo is used to identify Chinese Numbers from speech signals.       
---
### target
Recognition of Chinese eight-digit speech. / 识别中文的八位数字语音。

---

### main idea        
1. change speech signal to image         
2. use CNN recognize image

---
### plan A:  recognize-single-number   (Done)
- input:        
0_a.wav | 0_b.wav | 2_a.wav | 3_a.wav | 3_b.wav | ...    
- output:          
0  | 0 | 2 | 3 | 3 | ...      
- model：      
VGG13 for 10 class(0-9 numbers)
- data:          
2k+ single number specch data from 60+ people      
use 5 kinds of augment, include crop、loudness、noise、pitch and speed      
final training data 13k+         
- [x] pretrained_model      

---
### plan B:  end-2-end   (Lack of Data)
- input:        
08416923_a.wav | 79684315_b.wav | 29368741_a.wav | ...    
- output:          
08416923  | 79684315 | 29368741 |  ...      
- model：      
speech recognition has two models: acoustic-model and language-model       
plan-B just use acoustic-model: CNN+LSTM+CTC
- data:          
lack of continuous voice data         
- [ ] pretrained_model      