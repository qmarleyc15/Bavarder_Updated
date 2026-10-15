# Bavarder: fine-tuning small language models on French classroom conversations

An experiment in building a French conversation partner for new learners by fine-tuning small language models on transcribed spoken French. **This is not a finished chatbot.** The repo contains a working data pipeline, three training setups, and an honest comparison of how each one behaves, including where they fail and why.
## Motivation
I speak French and I’m interested in computational linguistics. When I was learning, I wished I had a conversation partner that used the French people actually speak. The French taught in my courses wasn’t really what people say day to day, so I spent a lot of time searching the internet for the everyday phrases that textbooks left out. This project asks whether a small language model fine-tuned on transcribed spoken French could fill that gap for new learners.
## Summary

- Built a parser that turns TRS/XML transcripts from the [TCOF corpus](https://doi.org/10.82270/tcof/v2.2) into clean child-adult conversation data pairs, with filters for minimal responses, near-echoes, and repeated phrases.
- Fine-tuned **BARThez** (French BART) and **Qwen2.5-1.5B-Instruct** with LoRA (under 1% of parameters trained) on a free Colab T4.
- Compared the untuned model, a single-turn fine-tune, and a multi-turn context fine-tune on the same prompts.
- **Result:** fine-tuning made replies sound like the corpus (casual spoken French) but mostly made them *less* relevant to the prompt. Adding previous turns as context gave modest, inconsistent gains. The remaining failures trace back to properties of the data, described below.

## Pipeline

```
TRS transcripts
   |  src/extract_pairs.py          parse XML, clean transcription markup, extract child -> adult pairs
   v
conversation_pairs.json            (4,744 pairs)
   |  src/echo_check.py             drop pairs where the adult mostly repeats the child
   |  src/cap_repeats.py            drop replies containing a 5-word phrase seen > 20 times
   v
single-turn data                   (4,528 pairs)  -> training/finetune_qwen.py, finetune_barthez.py

TRS transcripts
   |  src/build_context_pairs.py    previous 3 turns + adult reply, same filters built in
   v
context_pairs.json                 (4,528 examples from 69 transcripts) -> training/finetune_qwen_context.py
```

### Data decisions

- **Pairing rule:** an adult turn that directly follows a child turn. Other speaker roles are ignored.
- **Cleaning:** removes TCOF markup (pauses, liaison and hesitation markers, bracketed annotations, word fragments) and a few filler words.
- **Near-echo filter:** drops pairs where at least 70% of the adult's words already appear in the child's line (removed about 190 pairs).
- **Repetition cap:** one session in the corpus produced the same adult sentence many times, and both models memorized it. Replies containing a 5-word phrase that occurs more than 20 times are dropped.
- **Context windows:** up to 3 previous turns, each clipped to its last 60 words. Children are the denoted under the `user` role and adults the `assistant` role in the turn based chat template.
- **Evaluation split:** the context run splits by *source file*, so naturally no conversation appears in both train and test (neighboring examples share turns due to the added context, so a random split would leak).

## Repo layout

```
src/
  extract_pairs.py            TRS parser and pair extractor
  echo_check.py               echo-rate report and near-echo filter
  cap_repeats.py              repeated-phrase cap
  build_context_pairs.py      multi-turn context builder
training/
  finetune_barthez.py         BARThez + LoRA (single-turn)
  finetune_qwen.py            Qwen2.5-1.5B-Instruct + LoRA (single-turn)
  finetune_qwen_context.py    Qwen2.5-1.5B-Instruct + LoRA (context windows)
results/
  barthez.txt  qwen_single_turn.txt  qwen_context.txt
```

## Data

This repo contains **no corpus data**. Download the TCOF transcripts (version 2.2) from the corpus page: https://doi.org/10.82270/tcof/v2.2. The corpus is licensed under [CC BY-NC-SA 2.0](https://creativecommons.org/licenses/by-nc-sa/2.0/): non-commercial use only, with attribution and share-alike, so this project is for research and learning only. The transcripts include real children's first names, so I don't redistribute any derived data files either. Put the `.trs` files in `data/transcripts/` (this folder is git-ignored).

## Running it

I ran data processing locally and trained on Google Colab's free T4 GPU.

```
pip install -r requirements.txt

# 1. Extract conversation pairs
python src/extract_pairs.py data/transcripts -o conversation_pairs.json

# 2. Single-turn data: filter near-echoes, then put a cap on number of repeated phrases
python src/echo_check.py conversation_pairs.json --threshold 0.7 --out conversation_pairs_filtered.json
python src/cap_repeats.py conversation_pairs_filtered.json --out conversation_pairs_final.json

# 3. Context data: filters are baked in
python src/build_context_pairs.py data/transcripts --parser-module extract_pairs
```

Then upload the JSON file to Colab in the content folder (default upload location) and run the matching script in `training/`. Each script has its install commands in the header comments. Set the input filename at the top of the script. On a T4, the BARThez run took about 5 minutes to run 8 epochs  and each Qwen run 10 to 11 minutes to run 2 epochs.

## Results

Held-out perplexity value is **not comparable across runs**: the models use different tokenizers, and the splits and inputs differ.

| Run | Model | Input | Data | Split | Held-out perplexity |
|---|---|---|---|---|---|
| `barthez.txt` | BARThez + LoRA | "child"/user line  | 4,331 pairs (earlier cleaning) | random 90/10 | 10.92 |
| `qwen_single_turn.txt` | Qwen2.5-1.5B + LoRA | "child"/user line | 4,528 pairs | random 90/10 | 8.70 |
| `qwen_context.txt` | Qwen2.5-1.5B + LoRA | previous 3 turns | 4,528 examples | by transcript | 8.86 |

Full loss tables, settings, and sample generations are provided in `results/`. Samples come from a single sampled generation per prompt (the context run uses a fixed seed), so treat them as illustrations rather than measurements.

### What I found

1. **BARThez copies its input.** With default decoding, most outputs repeated all or part of the child’s line (in my first run, ‘Pourquoi le ciel est bleu ?’ came back unchanged). Filtering near-echoes removed only about 4% of the data and the copying continued, so it looks more like a property of the model than the dataset. Blocking copied phrases gave fluent but unrelated replies, so I switched to an instruction-tuned model, Qwen.
2. **Fine-tuning Qwen matched the register but lost relevance.** The untuned model answers the question (sometimes with errors in its French which was originally a feature to make learners aware of familiar French). The tuned model sounds like spoken French from the corpus but often replies with something unrelated.
3. **Context helped a little.** The context model gave on-topic replies in a few cases (a follow-up about vacations after a child described a weekend at the pool) but still drifted off-topic on others. Six prompts and one sample each is not enough to claim a reliable improvement.
4. **Cleaning bugs mattered.** A regex meant to strip word fragments was also deleting the "est-" in "est-ce que," leaving corrupted text that showed up in model output. A single recurring adult sentence was memorized by both model families until I capped repeated phrases.

## Limitations

- **The data is group classroom discussion, not one adult talking to one child.** About 14% of adult replies contain "vous", and replies often mix discussion with classroom logistics. About 6.5% of replies contain the word "question(s)", spread across 56 of the 69 transcripts, so the model keeps drifting toward questions about questions. So the corpus looks like many sessions of the same kind of discussion.
- **All children are merged into one speaker.** Consecutive child turns from different kids look like one person speaking.
- **It models how adults talk to children,** not how a tutor talks to an adult learner. (Which may function as a feature if used  in composition with other datasets since the original aim of the project was to jumpstart young French learners)
- **Small data.** About 4,500 examples from 69 transcripts, with long, rambling adult replies (mean 12 words, up to 100).
- **Weak evaluation.** Perplexity plus six hand-picked prompts. No human evaluation; given how off-topic the results were, it wasn’t worth it yet.
- **The untuned Qwen 1.5B model also makes French errors,** so it isn't a reliable baseline for a learner-facing tool either which is why it was not my first option.

## Possible next steps

- Label individual children as separate speakers in the context.
- Build an evaluation set with relevance ratings, and generate several samples per prompt.
- Try a larger French-capable instruction model, or instruction-style data written for a tutoring task.
- Add grammar error correction, the original goal, as a separate component.

## Acknowledgements

Transcripts from the TCOF corpus, version 2.2 (published 9 April 2024), https://doi.org/10.82270/tcof/v2.2, licensed under CC BY-NC-SA 2.0. See the corpus page for its authors and recommended citation. Built with Hugging Face `transformers` and `peft`.

## License

Code: MIT (see `LICENSE`). The TCOF corpus is not included and is licensed separately under CC BY-NC-SA 2.0. Sample model outputs in `results/` are derived from the corpus and are shared under the same CC BY-NC-SA 2.0 terms, so please don't use them commercially.
