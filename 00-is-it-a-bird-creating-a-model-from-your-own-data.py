# %% [markdown]
# ## Is it a bird?

# %% [markdown]
# In 2015 the idea of creating a computer system that could recognise birds was considered so outrageously challenging that it was the basis of [this XKCD joke](https://xkcd.com/1425/).

# %% [markdown]
# But today, we can do exactly that, in just a few minutes, using entirely free resources!
#
# The basic steps we'll take are:
#
# 1. Use DuckDuckGo to search for images of "bird photos"
# 1. Use DuckDuckGo to search for images of "forest photos"
# 1. Fine-tune a pretrained neural network to recognise these two groups
# 1. Try running this model on a picture of a bird and see if it works.

# %% [markdown]
# ## Step 1: Download images of birds and non-birds

# %%
import re
import time

from fastcore.foundation import L
from fastcore.net import urljson, urlread


def search_images(term: str, max_images: int = 200):
    """
    Search images on the web
    """
    url = "https://duckduckgo.com/"
    res = urlread(url, data={"q": term})
    search_object = re.search(r"vqd=([\d-]+)\&", res)
    if not search_object:
        raise ValueError(f"Could not find any result for {term}")

    request_url = f"{url}i.js"
    search_params = dict(l="us-en", o="json", q=term, vqd=search_object[1], f=",,,", p="1", v7exp="a")

    urls_set, data = set(), {"next": 1}
    while len(urls_set) < max_images and "next" in data:
        data = urljson(request_url, data=search_params)
        urls_set.update(L(data["results"]).itemgot("image"))
        request_url = url + data["next"]
        time.sleep(0.2)
    return L(urls_set)[:max_images]


# %% [markdown]
# Let's start by searching for a bird photo and seeing what kind of result we get. We'll start by getting URLs from a search:

# %%
urls = search_images("bird photos", max_images=1)
urls[0]

# %% [markdown]
# ...and then download a URL and take a look at it:

# %%
from fastdownload import download_url

dest = "data/bird.jpg"
download_url(urls[0], dest, show_progress=False)

from fastai.vision.all import Image

im = Image.open(dest)
im.to_thumb(256, 256)

# %% [markdown]
# Now let's do the same with "forest photos":

# %%
output_path = "data/forest.jpg"
download_url(search_images("forest photos", max_images=1)[0], output_path, show_progress=False)
Image.open(output_path).to_thumb(256, 256)

# %% [markdown]
# Our searches seem to be giving reasonable results, so let's grab 200 examples of each of "bird" and "forest" photos, and save each group of photos to a different folder:

# %%
from pathlib import Path
from fastai.vision.utils import download_images, resize_images

searches = "forest", "bird"
path = Path("bird_or_not")

for target in searches:
    dest = path / target
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f"{target} photo"))
    resize_images(path / target, max_size=400, dest=path / target)

# %% [markdown]
# ## Step 2: Train our model

# %% [markdown]
# Some photos might not download correctly which could cause our model training to fail, so we'll remove them:

# %%
from fastai.vision.utils import verify_images, get_image_files

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

# %% [markdown]
# To train a model, we'll need `DataLoaders`, which is an object that contains a *training set* (the images used to create a model) and a *validation set* (the images used to check the accuracy of a model -- not used during training). In `fastai` we can create that easily using a `DataBlock`, and view sample images from it:

# %%
from fastai.data.block import CategoryBlock, DataBlock
from fastai.data.transforms import RandomSplitter, parent_label
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock   

dls = DataBlock(
    blocks=[ImageBlock, CategoryBlock],
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method="squish")],
).dataloaders(path)

dls.show_batch(max_n=6)

# %% [markdown]
# Here what each of the `DataBlock` parameters means:
#
#     blocks=(ImageBlock, CategoryBlock),
#
# The inputs to our model are images, and the outputs are categories (in this case, "bird" or "forest").
#
#     get_items=get_image_files,
#
# To find all the inputs to our model, run the `get_image_files` function (which returns a list of all image files in a path).
#
#     splitter=RandomSplitter(valid_pct=0.2, seed=42),
#
# Split the data into training and validation sets randomly, using 20% of the data for the validation set.
#
#     get_y=parent_label,
#
# The labels (`y` values) is the name of the `parent` of each file (i.e. the name of the folder they're in, which will be *bird* or *forest*).
#
#     item_tfms=[Resize(192, method='squish')]
#
# Before training, resize each image to 192x192 pixels by "squishing" it (as opposed to cropping it).

# %% [markdown]
# Now we're ready to train our model. The fastest widely used computer vision model is `resnet18`. You can train this in a few minutes, even on a CPU! (On a GPU, it generally takes under 10 seconds...)
#
# `fastai` comes with a helpful `fine_tune()` method which automatically uses best practices for fine tuning a pre-trained model, so we'll use that.

# %%
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

# %% [markdown]
# Generally when I run this I see 100% accuracy on the validation set (although it might vary a bit from run to run).
#
# "Fine-tuning" a model means that we're starting with a model someone else has trained using some other dataset (called the *pretrained model*), and adjusting the weights a little bit so that the model learns to recognise your particular dataset. In this case, the pretrained model was trained to recognise photos in *imagenet*, and widely-used computer vision dataset with images covering 1000 categories) For details on fine-tuning and why it's important, check out the [free fast.ai course](https://course.fast.ai/).

# %% [markdown]
# ## Step 3: Use our model (and build your own!)

# %% [markdown]
# Let's see what our model thinks about that bird we downloaded at the start:

# %%
is_bird, _, probs = learn.predict(PILImage.create("bird.jpg"))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")

# %% [markdown]
# Good job, resnet18. :)
#
# So, as you see, in the space of a few years, creating computer vision classification models has gone from "so hard it's a joke" to "trivially easy and free"!
#
# It's not just in computer vision. Thanks to deep learning, computers can now do many things which seemed impossible just a few years ago, including [creating amazing artworks](https://openai.com/dall-e-2/), and [explaining jokes](https://www.datanami.com/2022/04/22/googles-massive-new-language-model-can-explain-jokes/). It's moving so fast that even experts in the field have trouble predicting how it's going to impact society in the coming years.
#
# One thing is clear -- it's important that we all do our best to understand this technology, because otherwise we'll get left behind!

# %% [markdown]
# Now it's your turn. Click "Copy & Edit" and try creating your own image classifier using your own image searches!
#
# If you enjoyed this, please consider clicking the "upvote" button in the top-right -- it's very encouraging to us notebook authors to know when people appreciate our work.
