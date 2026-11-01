# Speaker Script: Pretrained Networks and Transfer Learning

Target: about 18 minutes, leaving 2 minutes for questions.
*[Stage directions in brackets are for you, not to read aloud.]*

---

## Opening (≈ 0:45)

Hi everyone. Today I'm going to talk about pretrained networks and transfer learning.

I've put everything from the three notebooks into a review document that you can use to look things up later. Today I'll walk you through the main ideas and skip some of the details, but they're all in there if you want to read up on them.

Here's the plan: we'll start with where pretrained models come from, then look inside two of them, and then see how we can reuse them for our own problems, even when we only have very few images.

---

## 1. ImageNet (≈ 1:15)

*[Section 1]*

It all starts with ImageNet. ImageNet is a huge collection of labelled photos. For its yearly competition, the organisers used about **1.2 million training images in 1,000 classes**, from goldfish to umbrella.

In 2011, before CNNs entered the competition, the best models still got about a quarter of the images wrong. By 2015, CNNs had brought that error below 4 %.

*[point at the picture]*

The important part for us: the research groups didn't just publish papers. They published their networks **together with the learnt weights**. So today, anyone can download these networks with one line of code, for example with `keras.applications`, and use them right away.

*[point at the table]*

Here are a few of them. VGG16 is the classic big one, MobileNetV2 is the small, fast one, and there are many more.

---

## 2. Why Not Start From Scratch? (≈ 1:30)

*[Section 2]*

So why don't we just train our own CNN every time?

A CNN starts with random filters. To learn edges, textures and shapes on its own, it needs millions of labelled images, powerful GPUs, and days or weeks of training.

But most real problems don't have millions of images. You might have a few hundred photos of plant diseases, or sea animals, or defects in a factory.

The good news: a network trained on ImageNet has already learnt to recognise edges, textures, shapes and object parts. And these things are useful for almost any image problem. So we reuse them.

This idea is called **transfer learning**: a model trained for one task becomes the starting point for a related task.

Here's a way to think about it: someone who already speaks five languages learns a sixth one much faster than a child learns its first. A pretrained CNN has already learnt how to *see*. We only need to teach it a new task.

---

## 3. VGG16 and MobileNetV2 (≈ 4:00)

*[Section 3, VGG16 picture]*

Let's look inside two of these networks. First, **VGG16**.

VGG16 is the classic deep CNN. In blue you see the convolutional base: 13 convolutional layers in five blocks. The image gets smaller from left to right, and the number of channels grows. Then, in grey, `Flatten`, and in green, three large dense layers.

And the end is where VGG16 has its weakness. `Flatten` turns the last feature maps into 25,088 numbers, and every single one of them is connected to each of 4,096 neurons. That one layer holds **74 % of all 138 million parameters**. The convolutional layers, which actually find the features, hold only 11 %.

I like to think of VGG16 as a company. It has 13 departments that do the real work, and they all use the same simple method: they look at small 3 × 3 patches. But at the end sits a board of 4,096 members, and every member reads all 25,088 individual notes instead of a summary. That board meeting eats three quarters of the budget.

*[pause]*

Now **MobileNetV2**. It reaches a similar accuracy with only **3.5 million parameters**, that's 39 times fewer. It uses three ideas.

**First: depthwise separable convolutions.** A normal convolution does two things at once: it filters in space, and it mixes the channels. MobileNet splits that into two steps. First, every channel is filtered on its own. Then the channels are mixed, pixel by pixel. That needs about **8 times fewer** parameters.

*[point at panel A: red vs. green]*

*[Optional, skip if short on time: the recording studio analogy is in the document.]*

**Second: inverted residual blocks.** Each block first expands the channels, for example from 24 to 144, then filters them cheaply, and then compresses them back to 24.

It's like packing a suitcase. You unpack everything onto the bed, sort through each item, and pack it back into the small suitcase. Only the small suitcase travels from block to block.

*[point at the orange arrow in panel B]*

And there's a **shortcut**: the input of the block is added to its output. So the block only adds changes, and nothing gets lost. This also helps the training signal travel through the deep network.

**Third: global average pooling.** Remember the expensive board meeting in VGG16? MobileNetV2 replaces it with a one-page summary. Each of the 1,280 final feature maps is averaged to a single number, and then one dense layer makes the decision.

*[point at the table]*

So in short: VGG16 is big and expensive, especially at the end. MobileNetV2 is lean: 14 instead of 528 megabytes, and small enough to run on a phone.

---

## 4. Classifying a Photo in Four Steps (≈ 1:30)

*[Section 4]*

How do we actually use one of these networks? It takes four steps.

**One:** resize the photo to the size the network was trained on, for example 224 by 224 pixels.

**Two:** add a batch axis. Keras always expects a batch of images, even if you only have one.

**Three:** preprocess the photo with the network's own `preprocess_input` function. For MobileNetV2, that scales the pixel values from 0 to 255 down to minus 1 to 1.

**Four:** predict, and decode the 1,000 probabilities into class names.

*[point at "Step 3 is easy to forget"]*

Step three is the dangerous one, because if you forget it, **nothing crashes**. With preprocessing, MobileNetV2 correctly sees a llama. Without it, the same photo becomes a shower curtain.

It's like telling someone "it's 80 degrees outside" without saying Fahrenheit. The number is right, the unit is wrong, and so is the reaction.

---

## 5. Early and Late Layers (≈ 1:15)

*[Section 5, picture]*

What does a network actually see inside? We can find out by cutting it after an inner layer and looking at the feature maps.

In the **early layers**, the feature maps still look like filtered versions of the photo. They react to edges, lines, colours and textures.

In the **late layers**, the maps are small and mostly dark, with a few bright spots. They no longer show edges, but larger structures and parts of objects.

It's like learning to read: first strokes, then letters, then words, then meaning. The deeper the layer, the less it looks like the photo, and the more it means.

And nobody programmed these detectors. They emerged on their own during training.

---

## 6. A New Use Case: The Bean Leaf (≈ 1:00)

*[Section 6]*

So what happens if we show the network something that isn't in ImageNet? For example, a bean leaf with a plant disease.

MobileNetV2 says: **"sea urchin"**. With a probability of only 13 %, so it's very unsure.

But look closer: the answer isn't random. The spotted leaf really does look a bit like a sea urchin. So the feature recognition works. **Only the words are missing.**

Think of an experienced zoologist looking at the leaf and saying "that looks like sea urchin skin". They don't need to learn how to see. They only need to learn the new names.

So what do we do? We keep the part that sees, and we replace the part that names.

---

## 7. Transfer Learning in Practice (≈ 2:30)

*[Section 7, picture]*

That's exactly what transfer learning does. A pretrained CNN has two parts. The **convolutional base**, in blue, turns an image into features. The **classifier on top** turns those features into the 1,000 ImageNet classes.

We **keep the base and freeze it**, and we **replace the classifier** with a small new one for our own classes.

*[point at the code]*

In code, that's two lines. `include_top=False` removes the ImageNet classifier. `pooling="avg"` turns every image into 1,280 numbers. And `trainable = False` freezes the base.

**Why freeze?** Our new head starts with random weights, so at first it makes big mistakes. Without freezing, those mistakes would flow back into the base and damage everything it learnt on ImageNet.

**And why is this so fast?** A frozen base always turns the same image into the same 1,280 numbers. So we only need to run the base **once** over all our images. After that, we just train a small head on a table of numbers.

Think of a lab and a junior doctor. The lab analyses each patient's blood once and writes a report. The doctor then learns the diagnoses by reading the reports again and again. Nobody draws new blood for every study session.

*[point at the graphic]*

That's the difference here: about 1,000 passes through the base instead of 31,000.

And it works: in the notebook, one single dense layer on these features recognised bean leaf diseases with **91 % accuracy**, after a few seconds of training on about 1,000 photos.

---

## 8. Data Augmentation (≈ 1:00)

*[Section 8]*

With few images, the model sees every leaf in only one position. **Data augmentation** creates random variations during training: flips, rotations, zooms. A leaf upside down is still the same leaf, so the label stays correct, and the model overfits less.

But the variations have to be realistic. You can flip a dolphin horizontally, but not upside down.

The price: the images change in every epoch, so we can't compute the features only once anymore, and training gets slower.

One surprise you may see: the validation accuracy can be **higher** than the training accuracy. That's because augmentation and dropout are only active during training. It's like a runner who trains with a weight vest and then races without it.

---

## 9. Fine-Tuning (≈ 0:45)

*[Section 9]*

There's one more optional step: **fine-tuning**. After the new head is trained, we can unfreeze the top layers of the base and train them a little, so the features adapt to our task.

Three things to watch out for: train the head **first**, use a **very low learning rate**, and unfreeze only the **top** layers. The early layers are general, so they stay frozen.

---

## 10. What If There Are Very Few Images? (≈ 1:15)

*[Section 10]*

Finally, the big question: does this still work with really few images?

The exercise tested this with five sea animals, with **fewer than 60 training photos per class**.

The pretrained network on its own got only one of the five animals right. The dolphin, for example, became a great white shark, because ImageNet doesn't have a dolphin class.

But with transfer learning, the same frozen base plus a small new head reached about **92 % on validation and 96 % on the test set**. Augmentation didn't add anything here, so we chose the simpler, faster model.

And the few mistakes it made were all between crayfish, lobster and crab, which honestly look pretty similar.

So yes: transfer learning works even with a few dozen images per class, because the pretrained base already brings most of the knowledge.

---

## Summary (≈ 1:00)

*[Summary]*

Let me wrap up with five things to remember.

**One:** pretrained networks bring features learnt from more than a million photos.

**Two:** prepare your photos exactly as in training: the right size, a batch axis, and the network's own preprocessing.

**Three:** transfer learning means keeping the frozen base and training a new head. A few hundred photos are enough.

**Four:** with a frozen base, the features are computed once, so training takes seconds. Augmentation helps against overfitting, but it costs time.

**Five:** choose your model on the validation set, and use the test set only once. On small datasets, a single photo can make the difference.

Or in one sentence: **the pretrained network gives us the eyes. We only teach it the new words.**

Everything I skipped is in the document. Thank you, and I'm happy to take questions.

---

## Backup: Possible Questions

| Question | Short answer |
|---|---|
| Why 224 × 224? | It's the input size the network was trained on. MobileNetV2 also accepts 160 × 160, which is faster. |
| What does `None` mean in a shape? | The batch size, any number of images. |
| Why is the network's own `preprocess_input` needed? | Every network was trained with its own pixel scaling; VGG16 and MobileNetV2 use different ones. |
| Why not fine-tune right away? | The random new head would send large gradients into the base and destroy the pretrained filters. |
| Is a better ImageNet model always better for my task? | No. In notebook 02, EfficientNetB0 is better on ImageNet but gave worse features for bean leaves (0.85 vs. 0.91). |
| Why does test accuracy sometimes beat validation? | With 67 images, one photo is 1.5 %. A difference of two photos is chance. |
