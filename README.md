# Face_Recognition_System
Face recognition and biometrics are critical topics for various biomedical applications and numerous commercial use cases in daily life.

This project employs Autoencoder/Variational Autoencoder techniques for a FaceID system, covering the process from training to GUI application, using the DigiFace1M synthetic dataset. The application compares the encoded latent space of a face captured by a camera to the latent space of faces stored in a database. The model in this example was trained using a contrastive approach with triplet loss and MSE as the reconstruction loss.

Among the tested similarity measures (such as MSE, Cosine Similarity, and Pearson Correlation Coefficient), MSE demonstrated the best performance in distinguishing the same identity with and without glasses, as well as different identities.

<img src="graphics/me_glasses.png" alt="Database identity with glasses" width="300"/>
<img src="graphics/me_noglasses.png" alt="Database identity without glasses" width="300"/>
<img src="graphics/celebrity_glasses.png" alt="Identity out of database identity with glasses" width="300"/>
<img src="graphics/celebrity_noglasses.png" alt="Identity out of database identity without glasses" width="300"/>

