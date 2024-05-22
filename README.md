# Unmasking Privacy: A Reproduction and Evaluation Study of Obfuscation-based Perturbation Techniques for Collaborative Filtering

This repository contains code to reproduce our results from "Unmasking Privacy: A Reproduction and Evaluation Study of Obfuscation-based Perturbation Techniques for Collaborative Filtering" by Alex Martinez, Mihnea Tufis and Ludovico Boratto published at the The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR'24). 

# Abstract

Recommender systems (RecSys) solve personalisation problems and therefore heavily rely on personal data – demographics, user preferences, user interactions – each baring important privacy risks. It is also widely accepted that in RecSys performance and privacy are at odds, with the increase of one resulting in the decrease of the other. Among the diverse approaches in privacy enhancing technologies (PET) for RecSys, perturbation stands out for its simplicity and computational efficiency. It involves adding noise to sensitive data, thus hiding its real value from an untrusted actor. We reproduce and test a set of four randomization-based perturbation techniques developed by Batmaz and Polat [1] for privacy preserving collaborative filtering. While the framework presents great advantages – low computational requirements, several useful privacy-enhancing parameters – the supporting paper lacks conclusions drawn from empirical evaluation. We address this shortcoming by proposing – in absence of an implementation by the authors – our own implementation of the obfuscation framework. We then develop an evaluation framework to test the main assumption of the reference paper – that RecSys privacy and performance are competing goals. We extend this study to understand how much we can enhance privacy, within reasonable losses of the RecSys performance. We reproduce and test the framework for the more realistic scenario where only implicit feedback is available, using two well-known datasets (MovieLens-1M and Last.fm-1K), and several state-of-the-art recommendation algorithms (NCF and LightGCN from the Microsoft Recommenders public repository).

[1] Zeynep Batmaz and Huseyin Polat. 2016. Randomization-based privacy-preserving frameworks for collaborative filtering. Procedia Computer Science 96 (2016), 33–42

# Reproducibility
1. Download the datasets ([Movielens-1M](https://grouplens.org/datasets/movielens/1m/) and [Last.fm-1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)) and save them in ```\data```.
2. Run ```pre-processing.ipynb``` pre-process the dataset (as described in Section 2.1 of our paper) and create the train and test splits.
3. 
