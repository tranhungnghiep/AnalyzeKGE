# Analyzing Knowledge Graph Embedding Methods

This is a pure Python implementation of knowledge graph embedding (KGE) methods in TensorFlow 1.x/Keras, which was part of our experiments to unify previous KGE models under the perspective in our [DSI4 EDBT/ICDT 2019 paper](https://arxiv.org/abs/1903.11406), as well as a preliminary to our [ECAI 2020 paper](https://arxiv.org/abs/2006.16365) and [IJCAI 2022 paper](https://arxiv.org/abs/2209.15597). The codes demonstrate several important techniques in KGE and feature some recent state-of-the-art models including TransE, DistMult, CP, SimplE, ComplEx, and our proposed [Quaternion embeddings](https://arxiv.org/abs/1903.11406).

Knowledge graph embedding methods aim to learn low-dimensional vector representations of entities and relations in knowledge graphs. The models take input in the format of triples (h, t, r) denoting head entity, tail entity, and relation, respectively, and output their embedding vectors as well as solving link prediction. For more information, please see our paper.

## How to run
```shell script
python main.py --seed 7 --gpu 0 --model DistMult --in_path ../datasets/wn18/ --D 400 --Ce 1 --Cr 1 --sampling negsamp --batch_size 128 --neg_ratio 5 --max_epoch 500 --lr 1e-3 --lr_decay 1.0 --lmbda_ent 1e-4 --lmbda_rel 1e-4 --reg_n3 0 --constraint "" --to_constrain ""
```
```shell script
python main.py --seed 7 --gpu 0 --model CP --in_path ../datasets/wn18/ --D 200 --Ce 2 --Cr 2 --sampling negsamp --batch_size 128 --neg_ratio 5 --max_epoch 500 --lr 1e-3 --lr_decay 1.0 --lmbda_ent 1e-4 --lmbda_rel 1e-4 --reg_n3 0 --constraint "" --to_constrain ""
```
```shell script
python main.py --seed 7 --gpu 0 --model SimplE --in_path ../datasets/wn18/ --D 200 --Ce 2 --Cr 2 --sampling negsamp --batch_size 128 --neg_ratio 5 --max_epoch 500 --lr 1e-3 --lr_decay 1.0 --lmbda_ent 1e-4 --lmbda_rel 1e-4 --reg_n3 0 --constraint "" --to_constrain ""
```
```shell script
python main.py --seed 7 --gpu 0 --model ComplEx --in_path ../datasets/wn18/ --D 200 --Ce 2 --Cr 2 --sampling negsamp --batch_size 128 --neg_ratio 5 --max_epoch 500 --lr 1e-3 --lr_decay 1.0 --lmbda_ent 1e-4 --lmbda_rel 1e-4 --reg_n3 0 --constraint "" --to_constrain ""
```
```shell script
python main.py --seed 7 --gpu 0 --model Quaternion --in_path ../datasets/wn18/ --D 100 --Ce 4 --Cr 4 --sampling negsamp --batch_size 128 --neg_ratio 5 --max_epoch 500 --lr 1e-3 --lr_decay 1.0 --lmbda_ent 1e-4 --lmbda_rel 1e-4 --reg_n3 0 --constraint "unitnorm" --to_constrain "rowrel"
```

## How to cite
If you found this code or our work useful, please cite us.
- *Hung-Nghiep Tran and Atsuhiro Takasu. [Analyzing Knowledge Graph Embedding Methods from a Multi-Embedding Interaction Perspective](https://arxiv.org/abs/1903.11406). In Proceedings of DSI4 at EDBT/ICDT, 2019.*  
  ```
  @inproceedings{tran_analyzingknowledgegraph_2019,
    title = {Analyzing {Knowledge} {Graph} {Embedding} {Methods} from a {Multi}-{Embedding} {Interaction} {Perspective}},
    booktitle = {Proceedings of the {Data} {Science} for {Industry} 4.0 {Workshop} at {EDBT}/{ICDT}},
    author = {Tran, Hung-Nghiep and Takasu, Atsuhiro},
    year = {2019},
    pages = {7},
    url = {https://arxiv.org/abs/1903.11406},
  }
  ```
- *Hung-Nghiep Tran and Atsuhiro Takasu. [Multi-Partition Embedding Interaction with Block Term Format for Knowledge Graph Completion](https://arxiv.org/abs/2006.16365). In Proceedings of the European Conference on Artificial Intelligence (ECAI), 2020.*  
  ```
  @inproceedings{tran_multipartitionembeddinginteraction_2020,
    title = {Multi-{Partition} {Embedding} {Interaction} with {Block} {Term} {Format} for {Knowledge} {Graph} {Completion}},
    booktitle = {Proceedings of the {European} {Conference} on {Artificial} {Intelligence}},
    author = {Tran, Hung-Nghiep and Takasu, Atsuhiro},
    year = {2020},
    pages = {833--840},
    url = {https://arxiv.org/abs/2006.16365},
  }
  ```
- *Hung-Nghiep Tran and Atsuhiro Takasu. [MEIM: Multi-partition Embedding Interaction Beyond Block Term Format for Efficient and Expressive Link Prediction](https://arxiv.org/abs/2209.15597). In Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI), 2022.*  
  ```
  @inproceedings{tran_meimmultipartitionembedding_2022,
    title = {{MEIM}: {Multi}-partition {Embedding} {Interaction} {Beyond} {Block} {Term} {Format} for {Efficient} and {Expressive} {Link} {Prediction}},
    booktitle = {Proceedings of the {Thirty}-{First} {International} {Joint} {Conference} on {Artificial} {Intelligence}},
    author = {Tran, Hung-Nghiep and Takasu, Atsuhiro},
    year = {2022},
    pages = {2262--2269},
    url = {https://arxiv.org/abs/2209.15597},
  }
  ```

## See also
- The complete development, MEI-KGE (Multi-partition Embedding Interaction model): https://github.com/tranhungnghiep/MEI-KGE
- KG20C, a scholarly knowledge graph benchmark dataset: https://github.com/tranhungnghiep/KG20C
