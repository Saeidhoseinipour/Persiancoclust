
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

def generate_grid_of_wordclouds(Freq_block):
    ##### # Size image and mask

    configuration = {
        "language": "Urdu"
    }
    reshaper = ArabicReshaper(configuration=configuration)



    x, y = np.ogrid[:300, :300]
    mask = (x - 150) ** 2 + (y - 150) ** 2 > 140 ** 2
    mask = 255 * mask.astype(int)
    fig, axs = plt.subplots(8, 8, figsize=(20, 20))
    axs = axs.ravel()
    
    for i, ax in enumerate(axs):
        wordcloud = WordCloudFa(background_color="white",contour_color="red",font_path='D:/My paper/Application/NMTFcoclust/NotoNaskhArabic-Regular.ttf',no_reshape=False,colormap=plt.cm.gray,repeat=False,mask=mask,prefer_horizontal= 1.00).generate_from_frequencies(Freq_block[i])
        ax.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig("Wordcloud_8_8_Persian_news.png", format="png", bbox_inches="tight")
    plt.show()