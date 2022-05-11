import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
stop_words = stopwords.words("russian")
import pymorphy2
from summa import summarizer
from lexrank import STOPWORDS, LexRank
import pandas as pd
import razdel
import torch
from transformers import AutoTokenizer, \
                         BertForTokenClassification, \
                         AutoModelForCausalLM, \
                         MBartTokenizer, \
                         MBartForConditionalGeneration, \
                         T5ForConditionalGeneration

print('\033[92mImports done')


# ----------- Метод Луна ---------------
def preprocess(text):
    # токенизируем текст на уровне слов, убирая пунктуационные знаки
    tokenized = [[word.lower() for word in word_tokenize(sent) if has_letter_or_digit(word.lower())] \
                 for sent in sent_tokenize(text)]

    # лемматизируем текст
    morph = pymorphy2.MorphAnalyzer()
    lemmatized = [[morph.parse(word)[0].normal_form for word in sent] for sent in tokenized]

    return lemmatized


def has_letter_or_digit(string):
    for symbol in string:
        if symbol in 'йцукенгшщзхъфывапролджэячсмитьбюёqwertyuiopasdfghjklzxcvbnm0123456789':
            return True
    return False


def get_frequencies(preprocessed_text):
    frequencies = {}
    for sent in preprocessed_text:
        for token in sent:
            if token in frequencies.keys():
                frequencies[token] += 1
            else:
                frequencies[token] = 1

    # сортируем по убыванию частотности
    sorted_tuples = sorted(frequencies.items(), key=lambda x: x[1])[::-1]
    return {key: value for key, value in sorted_tuples}


def calculate_significance(sentence, main_words):
    # считаем количество значимых слов в предложении
    significant_words_count = 0
    for word in sentence:
        if word in main_words:
            significant_words_count += 1

    # считаем итоговую значимость
    if len(sentence) == 0:
        return 0

    significance = (significant_words_count ** 2) / len(sentence)
    return significance


def luhn_summarize(text, percentage=10):
    # предобработка текста
    preprocessed = preprocess(text)

    # формируем частотный словарь
    frequencies = get_frequencies(preprocessed)

    # значимые слова встречаются в тексте больше одного раза и не входят в список стоп-слов
    main_words = [key for key in frequencies.keys() if frequencies[key] > 1 and not key in stopwords.words("russian")]

    # считаем значимость для каждого предложения: (количество значимых слов ^ 2) / длина предложения
    significance_dict = {i: calculate_significance(sentence, main_words) for i, sentence in enumerate(preprocessed)}
    limit = round(len(preprocessed) * (percentage / 100))
    sorted_tuples_sign = sorted(significance_dict.items(), key=lambda x: x[1])[::-1][:limit]
    sentences_to_extract = sorted([sorted_tuple[0] for sorted_tuple in sorted_tuples_sign])

    # формируем реферат
    tokenized_sentences = sent_tokenize(text)
    summary = ''
    for idx in sentences_to_extract:
        summary += tokenized_sentences[idx] + ' '

    return summary[:-1]


print('\033[92mLuhn configured successfully')


# ------------ TextRank -------------
def get_formatted(text):
    return ' '.join(text.split('\n'))


def textrank_summarize(text, percentage=10):
    ratio = round(int(percentage) / 100, 2)
    summary = summarizer.summarize(text, ratio=ratio, language='russian')
    return get_formatted(summary)


print('\033[92mTextRank configured successfully')


# ----------- LexRank --------------
df = pd.read_csv('summarizer/data/df_gazeta_eval.csv')
lxr = LexRank(df['text'], stopwords=STOPWORDS['ru'])


def lexrank_summarize(text, percentage):
    tokenized = sent_tokenize(text)
    summary = lxr.get_summary(tokenized, summary_size=3, threshold=.1)
    return ' '.join(summary)


print('\033[92mLexRank configured successfully')


# ----------- BertSumExt ------------
model_name_bert = "IlyaGusev/rubert_ext_sum_gazeta"

tokenizer_bert = AutoTokenizer.from_pretrained(model_name_bert)
sep_token_bert = tokenizer_bert.sep_token
sep_token_id_bert = tokenizer_bert.sep_token_id

model_bert = BertForTokenClassification.from_pretrained(model_name_bert)


def bertsumext_summarize(text, percentage):
    article_text = text
    sentences = [s.text for s in razdel.sentenize(article_text)]
    article_text = sep_token_bert.join(sentences)

    inputs = tokenizer_bert(
        [article_text],
        max_length=500,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    sep_mask = inputs["input_ids"][0] == sep_token_id_bert

    # Fix token_type_ids
    current_token_type_id = 0
    for pos, input_id in enumerate(inputs["input_ids"][0]):
        inputs["token_type_ids"][0][pos] = current_token_type_id
        if input_id == sep_token_id_bert:
            current_token_type_id = 1 - current_token_type_id

    # Infer model
    with torch.no_grad():
        outputs = model_bert(**inputs)
    logits = outputs.logits[0, :, 1]

    # Choose sentences
    logits = logits[sep_mask]
    logits, indices = logits.sort(descending=True)
    logits, indices = logits.cpu().tolist(), indices.cpu().tolist()
    pairs = list(zip(logits, indices))
    pairs = pairs[:3]
    indices = list(sorted([idx for _, idx in pairs]))
    summary = " ".join([sentences[idx] for idx in indices])

    return summary


print('\033[92mBertSumExt configured successfully')


# ---------- GPT ------------
model_name_gpt = "IlyaGusev/rugpt3medium_sum_gazeta"
tokenizer_gpt = AutoTokenizer.from_pretrained(model_name_gpt)
model_gpt = AutoModelForCausalLM.from_pretrained(model_name_gpt)


def gpt_summarize(text, percentage):
    article_text = text

    text_tokens = tokenizer_gpt(
        article_text,
        max_length=600,
        add_special_tokens=False,
        padding=False,
        truncation=True
    )["input_ids"]
    input_ids = text_tokens + [tokenizer_gpt.sep_token_id]
    input_ids = torch.LongTensor([input_ids])

    output_ids = model_gpt.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4
    )

    summary = tokenizer_gpt.decode(output_ids[0], skip_special_tokens=False)
    summary = summary.split(tokenizer_gpt.sep_token)[1]
    summary = summary.split(tokenizer_gpt.eos_token)[0]

    return summary


print('\033[92mGPT configured successfully')


# ---------- mBART -------------
model_name_mbart = "IlyaGusev/mbart_ru_sum_gazeta"
tokenizer_mbart = MBartTokenizer.from_pretrained(model_name_mbart)
model_mbart = MBartForConditionalGeneration.from_pretrained(model_name_mbart)


def mbart_summarize(text, percentage):
    article_text = text

    input_ids = tokenizer_mbart(
        [article_text],
        max_length=600,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"]

    output_ids = model_mbart.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4
    )[0]

    summary = tokenizer_mbart.decode(output_ids, skip_special_tokens=True)

    return summary


print('\033[92mmBART configured successfully')


# ---------- T5 -------------
model_name_t5 = "IlyaGusev/rut5_base_sum_gazeta"
tokenizer_t5 = AutoTokenizer.from_pretrained(model_name_t5)
model_t5 = T5ForConditionalGeneration.from_pretrained(model_name_t5)


def t5_summarize(text, percentage):
    article_text = text

    input_ids = tokenizer_t5(
        [article_text],
        max_length=600,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]

    output_ids = model_t5.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=4
    )[0]

    summary = tokenizer_t5.decode(output_ids, skip_special_tokens=True)

    return summary


print('\033[92mT5 configured successfully')
