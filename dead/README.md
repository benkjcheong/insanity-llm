Loopymodel:
https://huggingface.co/TheBloke/MythoMax-L2-13B-GGUF/blob/main/mythomax-l2-13b.Q4_K_M.gguf
Use virtual environment & Nvidia EC2
Assumes scp of app.py into Nvidia EC2 initially before bash script is run; Requires high CPU & GPU usage
git clone https://github.com/benkjcheong/insanity-llm
I'm not patient enough; "pip3 install llama-cpp-python" takes too long

Finetuning:
In the future, upload small datasets via scp to EC2 where 2createmodel.py is run with Nvidia EC2.

- *Memoirs of My Nervous Illness* – **Daniel Paul Schreber**
- *To Have Done with the Judgment of God* – **Antonin Artaud**
- *Poems from the Tower Years* – **Friedrich Hölderlin**
- *The Trouble with Being Born* – **Emil Cioran**
- *Capitalist Realism: Is There No Alternative?* – **Mark Fisher**
- *The Work of Art in the Age of Mechanical Reproduction* – **Walter Benjamin**
- *Hyperion* – **Friedrich Hölderlin**
- *On the Heights of Despair* – **Emil Cioran**
- *The Theater and Its Double* – **Antonin Artaud**
- *The Peyote Dance* – **Antonin Artaud**
- *History and Utopia* – **Emil Cioran**
