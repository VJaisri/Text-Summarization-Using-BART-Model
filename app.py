from flask import Flask, render_template, request
from newspaper import Article
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')


@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', show_summary=False)

    elif request.method == 'POST':
        url = request.form['url']
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        title = article.title

        inputs = tokenizer([text], max_length=1024, return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=4000, early_stopping=True)
        summary = " ".join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

        return render_template('index.html', summary=summary, text=text, show_summary=True, title=title)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = True)
