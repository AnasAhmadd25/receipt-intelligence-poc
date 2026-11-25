from app.providers.ocr import reconstruct_lines

def test_word_boxes_preserve_product_row():
    words=[{'text':'12.50','left':100,'top':10,'width':30,'height':10,'conf':90,'block_num':1,'par_num':1,'line_num':1}, {'text':'Milk','left':10,'top':10,'width':30,'height':10,'conf':95,'block_num':1,'par_num':1,'line_num':1}, {'text':'Total','left':10,'top':40,'width':30,'height':10,'conf':91,'block_num':1,'par_num':1,'line_num':2}]
    rows=reconstruct_lines(words)
    assert len(rows)==2
    assert rows[0]['text']=='Milk 12.50'
    assert rows[0]['bbox']==[10,10,130,20]
