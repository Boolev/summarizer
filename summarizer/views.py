from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages

from .utils import luhn_summarize, \
                   textrank_summarize, \
                   lexrank_summarize, \
                   bertsumext_summarize, \
                   gpt_summarize, \
                   mbart_summarize, \
                   t5_summarize

sum_methods = {
    'Luhn': luhn_summarize,
    'TextRank': textrank_summarize,
    'LexRank': lexrank_summarize,
    'BertSumExt': bertsumext_summarize,
    'GPT': gpt_summarize,
    'mBART': mbart_summarize,
    'T5': t5_summarize,
}


def index(request):
    context = {}

    if request.method == 'POST':
        input_text = request.POST.get('input_text')
        method = request.POST.get('method')
        percentage = request.POST.get('percentage')

        if percentage == '':
            percentage = 10
        elif not 0 <= int(percentage) <= 100:
            percentage = 10
        else:
            percentage = int(percentage)

        summary = sum_methods[method](input_text, percentage=percentage)

        if len(summary) == 0:
            messages.warning(request, 'Попробуйте увеличить процент')

        context['input_text'] = input_text
        context['summary'] = summary
        context['selected_method'] = method

    return render(request, 'summarizer/index.html', context)
