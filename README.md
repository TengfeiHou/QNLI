## **QNLI Task**

update : This is a project I forked from another student in NLU class in SJTU. It cannot run. I asked author, but he not tell me. Perhaps according to the class rule it it not proper to share a project if it is a final project for a class.


I think it is a good idea to look at source code about [BERT](https://github.com/huggingface/pytorch-pretrained-BERT), not too hard to run this to get pass of class :) after finetune, you can easily get 90+ point in GLUE website. Good Luck boddy



----
the original README

#### **Download data**

You can use proxy in the script to download **QNLI** dataset from **GLUE**
`./pull_data.sh proxy_ip`
The default proxy is "https://127.0.0.1:1080"
`./pull_data.sh`

----

#### **Download Bert Models**

Run script
`./pull_data.sh proxy_ip`
or use default proxy "https://127.0.0.1:1080"
`./pull_data.sh`
The script will download all the following models to directory `data/.cache`

    bert-base-uncased
    bert-base-cased
    bert-large-uncased
    bert-large-uncased

----

#### **Make submission**

Move the target `QNLI.tsv` results to the directory `data/submission/` and run script
`./submit.sh`
This will overwrite and create `submission.zip` file in the root directory for submission.
