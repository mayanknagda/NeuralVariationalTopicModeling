import matplotlib.pyplot as plt
from wordcloud import WordCloud # pip install wordcloud
import torch
import pandas as pd
import numpy as np

def plot(beta, vocab, path, num_topics):
    dist = []
    f= open(path + "/topics.txt","w+")
    for i in range(beta.shape[0]):
        sorted_, indices = torch.sort(beta[i], descending=True)
        df = pd.DataFrame(indices[:30].numpy(), columns=['index'])
        dist.append(np.array(sorted_))
        names = pd.merge(df, vocab[['index', 'word']], how='left', on='index')['word'].values
        f.write(' '.join(names))
        f.write('\n')
        print(' '.join(names))
    f.close()
    filename = path +'/topics.txt'
    topics = []
    with open(filename,'r')as rf:
        for line in rf:
            words = line.strip().split()
            topics.append(words)
    topics_list = []
    for i in topics:
        a = ' '.join(i)
        topics_list.append(a)
    f=0.0
    for i,t in enumerate(topics):
        score=0.0
        for w in t:
            c=0.0
            for j,t2 in enumerate(topics):
                if i!=j:
                    for w2 in t2:
                        if w==w2:
                            c+=1
            score+=c/(len(topics)-1)
        score/=len(topics[0])#divide by number of words
        f+=score
    f/=len(topics)
    print(f)

    fig, axs = plt.subplots(int(num_topics/3 + 1), 3, figsize=(14, int(3.7 * num_topics/3))) # 7, 24
    for n in range(beta.shape[0]):
        i, j = divmod(n, 3)
        plot_word_cloud(beta[n], axs[i, j], vocab, n)
    axs[-1, -1].axis('off')
    plt.savefig(path + '/wordclouds')
    plt.close(fig)
    #return f, dist, topics_list


def plot_word_cloud(b, ax, vocab, n):
    sorted_, indices = torch.sort(b, descending=True)
    df = pd.DataFrame(indices[:100].numpy(), columns=['index'])
    words = pd.merge(df, vocab[['index', 'word']],
                     how='left', on='index')['word'].values.tolist()
    sizes = (sorted_[:100] * 1000).int().numpy().tolist()
    freqs = {words[i]: sizes[i] for i in range(len(words))}
    wc = WordCloud(background_color="black", width=800, height=500)
    wc = wc.generate_from_frequencies(freqs)
    ax.set_title('Topic %d' % (n + 1))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")