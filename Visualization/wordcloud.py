
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_wordcloud(dic_word_freq_1, dic_word_freq_2, dic_word_freq_3, dic_word_freq_4, dic_word_freq_5):
	##### # Size image and mask
	x, y = np.ogrid[:300, :300]
	mask = (x - 150) ** 2 + (y - 150) ** 2 > 160 ** 2
	mask = 255 * mask.astype(int)

                                        # fit function
	wc1 = WordCloud(background_color="white",contour_color="red", colormap='Reds',repeat=False, mask=mask,prefer_horizontal= 1.00)
	wc2 = WordCloud(background_color="white",contour_color="red", colormap='Reds',repeat=False, mask=mask,prefer_horizontal= 1.00)
	wc3 = WordCloud(background_color="white",contour_color="red", colormap='Reds',repeat=False, mask=mask,prefer_horizontal= 1.00)
	wc4 = WordCloud(background_color="white",contour_color="red", colormap='Reds',repeat=False, mask=mask,prefer_horizontal= 1.00)
	wc5 = WordCloud(background_color="white",contour_color="red", colormap='Reds',repeat=False, mask=mask,prefer_horizontal= 1.00)


	wc1 = wc1.generate_from_frequencies(dic_word_freq_1) 
	wc2 = wc2.generate_from_frequencies(dic_word_freq_2) 
	wc3 = wc3.generate_from_frequencies(dic_word_freq_3) 
	wc4 = wc4.generate_from_frequencies(dic_word_freq_4) 
	wc5 = wc5.generate_from_frequencies(dic_word_freq_5) 



                                        # Show

# size final image 
	fig=plt.figure(figsize=(20, 20), dpi=200)


# plot each image ...
# ... side by side
	fig.add_subplot(1, 5, 1)   # subplot 1
	plt.axis("off")
	plt.imshow(wc1, cmap=plt.cm.Reds, interpolation="bilinear")

	fig.add_subplot(1, 5, 2)   # subplot 2
	plt.axis("off")
	plt.imshow(wc2, cmap=plt.cm.Reds, interpolation="bilinear")

	fig.add_subplot(1, 5, 3)   # subplot 3
	plt.axis("off")
	plt.imshow(wc3, cmap=plt.cm.Reds, interpolation="bilinear")

	fig.add_subplot(1, 5, 4)   # subplot 3
	plt.axis("off")
	plt.imshow(wc4, cmap=plt.cm.Reds, interpolation="bilinear")

	fig.add_subplot(1, 5, 5)   # subplot 3
	plt.axis("off")
	plt.imshow(wc5, cmap=plt.cm.Reds, interpolation="bilinear")

	plt.show()

	plt.savefig("plot_wordcloud_Classic3.pdf", format="pdf")

	pass