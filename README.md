# SSViT-LRLPR — CNN–ViT official research implementation

Companion implementation for **“A CNN-ViT Framework for Recognizing Low-Resolution
License Plates”**, an OJITS manuscript under review. The proposed model combines a
topology-aware CNN spatial extractor, Surgical Focus Block (SFB), PixelShuffle
upsampling and a fixed-length cross-attention recognizer.

This clean repository preserves the validated model implementations, frozen
configurations and evaluation protocols. Model usability was checked against the
original checkpoints; full numerical experiments were not repeated for release.

## Setup

Use an isolated Python3.9.21 Linux environment. Install the original CUDA12.4
PyTorch build first, then the observed dependency constraints:

```bash
python -m pip install -c requirements/constraints-paper-py39.txt \
  torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -c requirements/constraints-paper-py39.txt -r requirements.txt
python -m unittest reproducibility.test_protocol reproducibility.test_smoke reproducibility.test_release reproducibility.test_consolidation reproducibility.test_infer -v
```

The constraints describe the working scientific environment; a fully independent
clean CUDA installation remains unverified. See [reader guidance](docs/READER_START_HERE.md).

## Original CNN–ViT checkpoint

[Download the original EMA checkpoint](https://github.com/valfride/SSViT-LRLPR/releases/download/ema-epoch58-v1.0.0/ema_weights.pth) from the
[single-model release](https://github.com/valfride/SSViT-LRLPR/releases/tag/ema-epoch58-v1.0.0). The uploaded asset was independently downloaded
and verified against the exact digest and byte size above.
[Weight access and mapping](docs/MODEL_WEIGHTS.md) distinguish distribution
authorization from the source-code license.

After obtaining the original file, verify it locally and run a bounded synthetic
single-image example. CPU FP32 is explicit here; this does not reproduce CUDA FP16
paper metrics. The checkpoint remains outside the checkout:

```bash
sha256sum /private/weights/ema_weights.pth
python -m reproducibility.infer \
  --checkpoint /private/weights/ema_weights.pth \
  --output /private/NEW-inference-output
```

Add `--image /private/authorized-crop.png` for your own authorized image. Output
predictions stay in the new private output directory. Use `--frames 3` or `--frames 5`
to repeat that one input for an interface demonstration; this is not real temporal
variation or a table evaluation. The four original fusion options are `bayes`
(**Product-Rule Fusion, PRF**), `average`, `majority` and `char_majority`.

## Baselines, data and reproducibility

Seven referenced recognizers are included with their native adapters: MDiff4STR,
CPPD, IGTR, LISTER, SVTRv2, OTE and SVTRv2-AR. The AR model is a controlled OpenOCR
SVTRNet/NRTR reconstruction, not the unavailable competition implementation.
See [baseline papers, implementations and adaptations](reproducibility/reference/BASELINE_PROVENANCE.csv)
and [source notices](NOTICE). Bibliographic citation is separate from software permission.
**No baseline, primary-model, other-seed or ablation checkpoints are distributed.**

Obtain datasets independently through their official access procedures:
[LRLPR-26](https://github.com/raysonlaroca/lrlpr-26-dataset) and
[LPLCv2](https://github.com/UFPR-IPASP-PR/LPLCv2-Dataset). No dataset images,
annotations, sample identifiers or plate transcriptions are included.

[Table commands](docs/COMMAND_RECIPES.md), [protocols](docs/PROTOCOLS.md) and
[paper/code mapping](docs/PAPER_TRACEABILITY.md) distinguish frozen inference,
prospective from-scratch training, three-seed ablations and efficiency measurement.
Exact original configurations,checkpoint/data hashes and native decoders are retained.
The proposed paper model uses **Cross-Entropy,AdamW,ReduceLROnPlateau,patience60
and EMA warmup5**. Legacy POLY configurations are historical, not the paper recipe.
The original continued training trajectory cannot be reconstructed completely.

Verify aggregate references without loading weights or datasets:

```bash
python -m reproducibility.verify --references --strict-paper-display
python -m reproducibility.run plan --table 1
```

Tables1–5 retain the original reported results. Table3 displays80.5%,67.7%,63.8%
remain separate from the exact integer-count arithmetic and approved historical
display disclosures. No checkpoints,results,selections or metrics were changed.
X3 remains deferred. Dataset and unavailable-weight access limits prevent claiming
that a fresh reader can independently reproduce every numerical table today.

## License and citation

The authors’ original contributions are licensed under [Apache-2.0](LICENSE).
Third-party files retain their applicable Apache/MIT terms, copyright notices and
local modification notices; see [NOTICE](NOTICE) and [third-party attribution](third_party/README.md).
The source license does not automatically license model weights or datasets.

[CITATION.cff](CITATION.cff) describes the software. The companion CNN–ViT manuscript
has no suitable self-entry in the reviewed manuscript bibliography, so no published
venue,DOI or BibTeX record is invented. An author-approved manuscript citation will
be added when available. The entries below come from the manuscript bibliography;
the competition entry’s final pages/DOI are corrected from
[the publisher record](https://link.springer.com/chapter/10.1007/978-3-032-31936-4_18).

## BibTeX References

```bibtex
@inproceedings{du2026mdiff4str,
  title={{MDiff4STR}: Mask Diffusion Model for Scene Text Recognition},
  author={Du, Yongkun and Zhao, Miaomiao and Fan, Songlin and Chen, Zhineng and Jia, Caiyan and Jiang, Yu-Gang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={5},
  pages={3705--3713},
  year={2026}
}

@article{du2025context,
  title={Context perception parallel decoder for scene text recognition},
  author={Du, Yongkun and Chen, Zhineng and Jia, Caiyan and Yin, Xiaoting and Li, Chenxia and Du, Yuning and Jiang, Yu-Gang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}

@article{du2025instruction,
  title={Instruction-guided scene text recognition},
  author={Du, Yongkun and Chen, Zhineng and Su, Yuchen and Jia, Caiyan and Jiang, Yu-Gang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={47},
  number={4},
  pages={2723--2738},
  year={2025},
  publisher={IEEE}
}

@inproceedings{cheng2023lister,
  title={{LISTER}: Neighbor decoding for length-insensitive scene text recognition},
  author={Cheng, Changxu and Wang, Peng and Da, Cheng and Zheng, Qi and Yao, Cong},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  pages={19541--19551},
  year={2023}
}

@inproceedings{du2025svtrv2,
  title={{SVTRv2}: {CTC} beats encoder-decoder models in scene text recognition},
  author={Du, Yongkun and Chen, Zhineng and Xie, Hongtao and Jia, Caiyan and Jiang, Yu-Gang},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  pages={20147--20156},
  year={2025}
}

@inproceedings{xu2024ote,
  title={{OTE}: Exploring accurate scene text recognition using one token},
  author={Xu, Jianjun and Wang, Yuxin and Xie, Hongtao and Zhang, Yongdong},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={28327--28336},
  year={2024}
}

@inproceedings{laroca2026competition,
  title = {{ICPR} 2026 {C}ompetition on Low-Resolution License Plate Recognition},
  author = {R. Laroca and V. Nascimento and D. Kim and S. Chung and S. Bae and U. Seo and S. Oh and C. M. Phung and M. G. Vo and X. Ye and Y. Du and Y. Su and Z. Chen and S. Heo and H. Lee and K. Na and K. V. Vu Nguyen and S. T. Pham and D. N. N. Phung and T. P. Le and Vy N. Vo Tran and D. Menotti},
  year = {2026},
  month = {Aug},
  booktitle = {International Conference on Pattern Recognition (ICPR)},
  pages = {256--275},
  doi = {10.1007/978-3-032-31936-4_18},
}

@inproceedings{sheng2019nrtr,
  author    = {Sheng, Fenfen and Chen, Zhineng and Xu, Bo},
  title     = {{NRTR}: A No-Recurrence Sequence-to-Sequence Model for Scene Text Recognition},
  booktitle = {International Conference on Document Analysis and Recognition (ICDAR)},
  pages     = {781--786},
  year      = {2019},
  doi       = {10.1109/ICDAR.2019.00130}
}
```
