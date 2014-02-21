##Decoder

####CIS526, Machine Translation, HW2

**Sean Welleck**

This project is related to decoding a source sentence by maximizing the probability of the target sentence.

The project contains three decoders:  
- Monotone Decoder  
- Stack Decoder  
- [Greedy Decoder](http://www.iro.umontreal.ca/~felipe/bib2webV0.81/cv/papers/paper-tmi-2007.pdf)  


And functions to combine two decodings.

Run ```python decode > output.txt``` to decode using the default input files, and output the translations to output.txt.

Run ```python combine -x filename1 -y filename2 > combined.txt``` to combine two decoded files, by choosing the higher scoring sentence, and output to combined.txt.  

-----
#####Algorithm
1. Decode with the monotone decoder.
2. Decode with the greedy decoder, using the decodings from (1) as the initial seed decoding.
3. Save decodings from (2).
4. Decode with the stack decoder.
5. Decode with the greedy decoder, using the decodings from (4) as the initial seed decoding.
6. Combine decodings from (5) and (2).

#####Other
1. Uses a combination of histogram pruning and threshold pruning.
2. Uses <= 40 translations per phrase.


-----
#####decoder.py
Contains the decoder implementations in a single ```Decoder``` class.

Contains the top-level user functions ```decode()``` and ```combine()```.
#####evaluator.py
Contains an ```Evaluator``` class that adapts the grading function.

Used to choose between two sentence translations while combining decodings.

Used as an alternative scoring function for the greedy decoder. Due to performance, I ended up just using my original, simpler scoring function.

