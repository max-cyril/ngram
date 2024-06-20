====================
N-gram model predict
====================


# ngram
Predict next word based on two words context

1- clone repository `ngram`

2- open terminal and enter into directory  ngram/ 

3- launch Docker if not & excecute 
        ```bash
        docker-compose up
        ```
4- to communicate with the API via streamlit Interface go to `localhost:8501` on your navigator


4'- to  directly use ngram API, go to `localhost:8000`


PS: if you got this following error at step 3 :
        ```
        Bind for 0.0.0.0:8001 failed: port is already allocated
        ```

    on ``docker-compose.yml``: replace 8001 by 8003  as folowing:

    ```
    services:
      ngram_api:
        build:
          context: ./scr/back/
          dockerfile: back.dockerfile
        ports:
          - "8001:8000"
    ```

    by

    ```
    services:
      ngram_api:
        build:
          context: ./scr/back/
          dockerfile: back.dockerfile
        ports:
          - "8002:8000"
    ``` 

    an rexecute ``docker-compose up``.

Lest Enjoy it!

    
    
