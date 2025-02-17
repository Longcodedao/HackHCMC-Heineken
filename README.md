# HackHCMC-Heineken

Heineken is looking to leverage image analysis to tackle key business and marketing challenges. Our project aims to develop a system that can automatically count the number of consumers at events and stores, a valuable metric for gauging campaign reach. Additionally, it will detect Heineken's billboard advertisements in the environment, allowing for better assessment of campaign placement. By analyzing footage from events, we can not only determine the number of participants but also gauge their emotions, providing insights into audience engagement. Furthermore, the system will track the presence and placement of Heineken's promotional materials (PGs/PBs) at various locations. This will be complemented by the ability to evaluate brand presence in stores by automatically identifying Heineken's promotional materials within images. This comprehensive approach will provide Heineken with valuable data to optimize marketing strategies and gain a deeper understanding of customer behavior.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Longcodedao/HackHCMC-Heineken.git
    ```
2. Create the Conda virtual environment and run the installation of the packages

   ```bash
    pip install -r requirements.txt
    ```
3. With the directory of the test images, we can run the inference.py file

   ```bash
    python inference.py --image_dir /path/towards_testing_images/
   ```

