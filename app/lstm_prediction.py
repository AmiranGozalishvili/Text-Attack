from keras_preprocessing.sequence import pad_sequences


# # 5 randomely selected reviews
# reviews = ["this dress is perfection! so pretty and flattering.",
#            "this is my new favorite top! looks and fits as described.",
#            "i could wear this every day, it is stylish and comfortable",
#            "material is too thin and quality is poor",
#            "it is nice material but the design makes you look like a pregnant lady"]
#


def model_pred(text, model, tokenizer, maxlen):
    """
      Use the trained LSTM to make predictions on new examples.
      """
    tokens = tokenizer.texts_to_sequences([text])
    tokens_pad = pad_sequences(tokens, maxlen=maxlen)
    # tokens_pad.shape
    model_pred1 = model.predict(tokens_pad)

    # predict = model.predict(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=maxlen))

    conf_val = model_pred1[0][0]
    result = None

    if conf_val >= 0.5:
        result = f"'{text}'\nRecommended | {int(conf_val * 100)}% Confidence\n"
    else:
        result = f"'{text}'\nNot Recommended | {int(conf_val * 100)}% Confidence\n"

    print("lstm prediction checkpoint")
    return result

# for i in reviews:
#     model_pred(i)
#
