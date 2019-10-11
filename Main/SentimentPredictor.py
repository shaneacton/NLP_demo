import torch

from Main.train_GloVe import load_imdb_for_glove, init_model, get_prediction_for_sentence

if __name__ == "__main__":
    load_imdb_for_glove()
    model, _ = init_model(False)
    model.load_state_dict(torch.load("tut2-model.pt"))

    print("model loaded")
    while True:

        inp = input("type a sentence:\n")
        if len(inp) ==0:
            break

        pred = get_prediction_for_sentence(model, inp)
        if pred.item() >= 0.5:
            print("positive", pred.item())
        elif pred.item() >= 0.4:
            print("neutral", pred.item())
        else:
            print("negative", pred.item())
