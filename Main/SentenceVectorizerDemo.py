from pytorch_pretrained_bert import BertTokenizer

from DataProcessor.BERT import prep_sentence_bert

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    while True:
        inp = input("type a sentence:\n")
        if len(inp) == 0:
            break

        print('Vectorized: ', prep_sentence_bert(inp, tokenizer))
