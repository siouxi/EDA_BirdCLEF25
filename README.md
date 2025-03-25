# 🐦 EDA_BirdCLEF25 🐦

# **Introduction**

https://www.kaggle.com/competitions/birdclef-2025 

Mobile and habitat-diverse species serve as key biodiversity indicators, as their population dynamics reflect ecological restoration success. However, traditional biodiversity surveys are costly and logistically complex. Passive acoustic monitoring (PAM), combined with machine learning, enables large-scale biodiversity assessments with higher temporal resolution, improving insights into restoration efforts.  

This competition focuses on developing machine learning models to identify species from acoustic data, especially those under-studied. Participants will create computational methods to process continuous audio and recognize species with minimal labeled data. Successful models will support biodiversity monitoring efforts, particularly in Colombia’s Magdalena Valley lowlands.  

The region is a biodiversity hotspot, yet over 70% of its rainforests have been lost due to cattle ranching and illegal logging. Fundación Biodiversa Colombia leads conservation efforts, including ecological restoration in El Silencio Natural Reserve. The competition aims to automate species detection, improve rare species classification, and leverage unlabeled data to enhance biodiversity assessments.  

The initiative is supported by multiple organizations, including Fundación Biodiversa Colombia, Google Research, and the Cornell Lab of Ornithology. Your contributions will aid conservationists in evaluating threats and refining restoration strategies effectively.

**I will divide the analysis into several sections, where I will break down the examination in detail.**  

# **Understandind Data**

The **BirdCLEF+ 2025** competition on Kaggle focuses on identifying species (birds, amphibians, mammals, and insects) from audio recordings collected in the El Silencio Natural Reserve, Colombia. This challenge is crucial for improving the monitoring of animal populations for conservation purposes.  

**Provided Files:**  

- `train_audio/`: Short recordings of individual species' sounds, contributed by users from xeno-canto.org, iNaturalist, and the Colombian Sound Archive (CSA) of the Instituto Humboldt.  

- `test_soundscapes/`: Approximately 700 one-minute recordings in ogg format at 32 kHz, used to evaluate the submitted solutions.  

- `train.csv`: Metadata associated with the training recordings, including:  
  - `primary_label`: Species code.  
  - `secondary_labels`: Other species present in the recording.
  - `scientific name`
  - `latitude & longitude`: Recording coordinates.  
  - `author`: User who provided the recording.  
  - `filename`: Audio file name.  
  - `rating`: Recording quality (1 to 5). 0 for no rating.
  - `collection`: Source of the recording (XC, iNat, or CSA).  

- `sample_submission.csv`: Example of the format for predictions, where each row represents an audio segment, and the columns correspond to the target species.  

- `taxonomy.csv`: Information about the species, including taxon ID and class (Aves, Amphibia, Mammalia, Insecta).  

The **scientific name** follows **binomial nomenclature**, which allows us to quickly identify the **genus** and the **epithet**. Therefore, we will use ``REGEX`` to split them into two columns, `genus` and `species`.


# **Basic analysis**

Thanks to the profiling done by ProfileReport, we can extract the following insights:  

- The classes are highly imbalanced.  
- Latitude has 809 missing values.  
- Longitude has 809 missing values.  
- Rating has approximately 7,948 zeros (27.8%), which means that a zero indicates no rating is available
-Incomplete scientific names

Most of the samples in the dataset are birds, which also group most of the samples that do have a valid response for the variable type, we must develop a strategy such as data augmentation for all other classes

<p align="center">
    <img src="plots/NumberSamplesPerClasses.png" width="600">
</p>


Thanks to separating the **genus** and the **epithet**, we can count how much information is available in each group. For example, even though birds are the most numerous group, some bird ``genus`` have very few samples.

This count will be recorded in four different ``CSV`` files, depending on the class: ``GenusAves.csv``, ``GenusInsecta.csv``, ``GenusMammalia.csv``, and ``GenusAmphibia.csv`` respectively. This will help us determine which ``genus`` and ``species`` to focus on if we decide to perform **data augmentation**, this count will be stored in the `count` variable.

 **Aves**

<p align="center">
    <img src="plots/AvesDistributionByEpithet.png" width="600">
</p>

 **Amphibia**

<p align="center">
    <img src="plots/AmphibiaDistributionByEpithet.png" width="600">

  </p>

   **Mammalia**

<p align="center">
    <img src="plots/MammaliaDistributionByEpithet.png" width="600">

  </p>

**Insecta**  
<p align="center">
    <img src="plots/InsectaDistributionByEpithet.png" width="600">
</p>


The families **Gryllidae, Cicadidae, and Tettigoniidae** group insects with unique characteristics within the orders **Orthoptera** and **Hemiptera**. **Gryllidae** includes crickets, known for their nocturnal chirping through stridulation. **Cicadidae** encompasses cicadas, famous for their loud sounds produced with tymbals and long life cycles. **Tettigoniidae**, which includes katydids or bush crickets, is distinguished by its camouflage and long antennae. Although they differ in morphology and communication methods, all play key roles in ecosystems and bioacoustics.

The previously mentioned **families** will be treated separately since they constitute a majority within the **Insecta** class. We will store this information in ``FamiliaeInsecta.csv``.

**Missing values:**

These values represent the **missing data** per class for ``latitude`` and ``longitude``:  

- **Aves** → **777** missing values  
- **Amphibia** → **30** missing values  
- **Mammalia** → **2** missing values  

This indicates how much geolocation data is absent for each class.

I have added information about the number of missing geolocation values to the previous ``CSV`` files by ``genus`` in the ``missing`` variable. Additionally, note that since **Insecta** has no missing values, neither ``GenusInsecta.csv`` nor ``FamiliaeInsecta.csv`` will include the ``missing`` variable.


 **Aves**


<p align="center">
    <img src="plots/AvesMissingsByEpithet.png" width="600">
</p>

 **Amphibia**

<p align="center">
    <img src="plots/AmphibiaMissingsByEpithet.png" width="600">

  </p>

   **Mammalia**

<p align="center">
    <img src="plots/MammaliaMissingsByEpithet.png" width="600">

  </p>


So we see that the data we can use in geographical terms is:

| Class      | Total Data | Missing Geographic Information | Complete Data | Percentage Loss |
|-----------|------------|--------------------------------|---------------|----------------|
| **Aves**      | 27,648     | 777                            | 26,871        | 2.81%          |
| **Amphibia**  | 583        | 30                             | 553           | 5.15%          |
| **Mammalia**  | 178        | 2                              | 176           | 1.12%          |

---
Additionally, there is a loss of ``rating``, which will likely require us to develop better measures for handling audio playback based on ``rating``.

| Taxon     | Total Data | Rating without score  | Usable Data | % Loss  |
|-----------|-----------|--------------------------|-------------|--------|
| **Aves**      | 27,648    | 7,196                    | 20,452      | 26.02% |
| **Amphibia**  | 583       | 436                      | 147         | 74.79% |
| **Mammalia**  | 178       | 161                      | 17          | 90.45% |


And **no Insecta has a rating score**

# **Geographical/spatial analysis**


<p align="center">
    <img src="ElSilencio.png">
</p>

It is assumed that the recordings were made in the 'El Silencio' nature reserve in Yondó, according to the *recording_location.txt* file. However, the coordinates of the audio files do not match this location. Therefore, it is necessary to question why this discrepancy occurs.

An interactive map is generated below and exported to HTML for visualization.

---

https://www.portal30x30.com/datasets/ada0fd60e00e4591a31db2c22ff82902_59/explore?filters=eyJOb21icmUiOlsiUmVzZXJ2YSBCaW9s82dpY2EgRWwgU2lsZW5jaW8iXX0%3D&location=6.572503%2C-70.298777%2C7.49

---


Using **KeplerGL**, I developed an **interactive map** to quickly visualize the distributions by class. This map is named ``Taxotrain.html``.


**Aves:**

<p align="center">
    <img src="maps/aves.png">
</p>

**Mammalia:**

<p align="center">
    <img src="maps/mammalia.png">
</p>

**Amphibia:**

<p align="center">
    <img src="maps/amphibia.png">
</p>

**Insecta:**

<p align="center">
    <img src="maps/insecta.png">
</p>


Birds are the most widely dispersed group, being present on nearly all continents, with a primary concentration in Central and South America. Mammals are mainly found in North and South America, while amphibians are predominantly located in South America. Insects also show a strong presence in South America, particularly in countries such as Colombia, Brazil, and Ecuador. Additionally, there are outlier insect data points near India; however, these do not significantly impact the analysis, as they belong to insect families within the broader taxonomic groups.


For each ``genus`` by ``species``, I calculated ``centroid_x``, ``centroid_y``, ``std_x``, ``std_y``, and ``radio_dis``—variables that I appended to each ``Genus`` ``CSV`` file as well as ``FamiliaeInsecta.csv``. This allowed me to identify highly dispersed groups and species, which could help us detect **dialects** among the animals themselves.


**Conclusions:**

Thanks to this analysis, we can identify anomalies in the dataset. One useful case is within the bird class, specifically the Ara genus. Its three species—Ara ararauna, Ara severus, and Ara chloropterus—exhibit distinct spatial distributions."

<p align="center">
    <img src="maps/ara.png">
</p>

And when divided by species, a clear spatial relationship becomes evident

<div style="display: flex; justify-content: center; gap: 10px;">
    <img src="maps/severus.png" style="width: 30%; height: auto;">
    <img src="maps/ararauna.png" style="width: 30%; height: auto;">
    <img src="maps/chloropterus.png" style="width: 30%; height: auto;">
</div>

With Ara severus, Ara ararauna, and Ara chloropterus showing distinct spatial distributions, respectively.

At other times, the analysis does not provide significant insights. By calculating the centroid and the mean distance, we can determine which groups are more dispersed; however, this does not necessarily aid in classification based on location. This is the case for the Pecari within the Mammalia group.

<p align="center">
    <img src="maps/pecari_mammalia.png">
</p>

In other cases, the analysis reveals patterns within the dataset, such as certain insect specimens that share identical coordinates while only differing in their audio recordings. This is exemplified by the insect Copris, i am also providing the generated data on geographical relationships in CSV format.

# **Audio Analysis**

Since the audio recordings contain expert voices, it becomes difficult to appreciate the species' sounds, or they may simply disappear. Therefore, it is necessary to identify the moments when they speak in order to determine the periods worth analyzing.

We achieve this using Silero's VAD model for Torch, which directly provides timestamps in seconds indicating when speech occurs. We then store this information in the variable `human_voices`, where each element is a list of dictionaries containing the start and end times of speech segments. Empty elements indicate periods when no one is speaking.

Thanks to ``Silero_VAD``, a **voice activity detection** model, we processed each audio file to obtain **timestamps** (in seconds) indicating where speech occurs.

I also created a function that takes the ``filename`` and a list of **timestamps**, then removes those segments from the audio file.I named this function `remove_audio_voices`.

