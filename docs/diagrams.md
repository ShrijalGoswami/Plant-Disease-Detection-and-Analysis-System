# Design Diagrams

All diagrams are written in Mermaid so they can be rendered on GitHub or pasted into
https://mermaid.live to export PNG images for the report. Each diagram describes the
code as it actually exists in this repository.

There is no ER diagram because the project has no database. Prediction history is a
Python list stored in Streamlit session state and is discarded when the browser tab
closes. The only files written to disk are produced by the training scripts
(weights, class list, metrics, confusion matrix).

## 1. System Architecture Diagram

```mermaid
flowchart LR
    subgraph UI["Presentation layer (app.py, Streamlit)"]
        U1[Detect tab]
        U2[Session analytics tab]
        U3[Model evaluation tab]
    end

    subgraph M1["Module 1: Image processing (src/preprocessing.py)"]
        P1[validate_upload] --> P2[decode_image] --> P3[resize_image] --> P4[normalize_image] --> P5[to_tensor]
    end

    subgraph M2["Module 2: Disease detection (src/model.py, src/prediction.py)"]
        D1[load_trained_model] --> D2[MobileNetV2]
        D2 --> D3[predict: softmax, top-k]
    end

    subgraph M3["Module 3: Analysis and reporting (src/analytics.py)"]
        A1[session history list] --> A2[summarize_history]
        A1 --> A3[class_distribution]
        A1 --> A4[build_text_report]
    end

    subgraph TR["Offline training (training/)"]
        T1[prepare_data.py] --> T2[train.py] --> T3[evaluate.py]
    end

    subgraph FS["Files on disk"]
        F1[(data/plantvillage)]
        F2[(data/splits/*.csv)]
        F3[(models/*.pt, class_names.json)]
        F4[(models/metrics.json, confusion_matrix.png)]
    end

    U1 --> P1
    P5 --> D3
    D3 --> U1
    D3 --> A1
    A2 --> U2
    A3 --> U2
    A4 --> U2
    F4 --> U3
    F1 --> T1
    T1 --> F2
    F2 --> T2
    T2 --> F3
    F3 --> D1
    F3 --> T3
    T3 --> F4
```

## 2. Workflow Diagram (user interaction)

```mermaid
flowchart TD
    S([Start app]) --> L{Trained model files present?}
    L -- no --> E1[Show error: run training/train.py] --> X([Stop])
    L -- yes --> W[Wait for image upload]
    W --> V{Extension and size valid?}
    V -- no --> E2[Show validation error] --> W
    V -- yes --> D{Decodes as image?}
    D -- no --> E3[Show corrupted file error] --> W
    D -- yes --> R[Resize to 224x224 and normalise]
    R --> SH[Show original and processed image]
    SH --> PR[Run MobileNetV2, softmax]
    PR --> C{Confidence below 60%?}
    C -- yes --> WN[Show low confidence warning]
    C -- no --> RES
    WN --> RES[Show crop, condition, status, confidence, top 3]
    RES --> H[Append entry to session history]
    H --> AN[Analytics tab: table, counts, charts, downloads]
    AN --> W
```

## 3. Training Workflow Diagram

```mermaid
flowchart LR
    A[data/plantvillage/class/image.jpg] --> B[prepare_data.py: stratified 70/15/15 split, seed 42]
    B --> C[(train.csv, val.csv, test.csv)]
    C --> D[train.py: augment train images, fine tune MobileNetV2, keep best val epoch]
    D --> E[(mobilenetv2_plant_disease.pt, class_names.json, training_history.json)]
    C --> F[evaluate.py: predict on test.csv]
    E --> F
    F --> G[(metrics.json, confusion_matrix.png, confusion_matrix.csv)]
```

## 4. Use Case Diagram

Mermaid has no native use case shape, so actors are shown as rounded nodes and use
cases as ellipses.

```mermaid
flowchart LR
    User((User))
    Dev((Developer / Student))

    UC1([Upload leaf image])
    UC2([View original and processed image])
    UC3([Get disease prediction with confidence])
    UC4([View session history and statistics])
    UC5([Download CSV history or text report])
    UC6([View model evaluation metrics])
    UC7([Prepare dataset split])
    UC8([Train model])
    UC9([Evaluate model on test set])
    UC10([Run unit tests])

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
    Dev --> UC7
    Dev --> UC8
    Dev --> UC9
    Dev --> UC10
    UC1 -. includes .-> UC2
    UC1 -. includes .-> UC3
    UC3 -. extends .-> UC4
```

## 5. Sequence Diagram (single prediction)

```mermaid
sequenceDiagram
    actor User
    participant App as app.py
    participant Pre as preprocessing.py
    participant Model as model.py
    participant Pred as prediction.py
    participant Ana as analytics.py

    User->>App: upload leaf.jpg
    App->>Pre: preprocess_upload(name, bytes)
    Pre->>Pre: validate_upload
    Pre->>Pre: decode_image (OpenCV, BGR to RGB)
    Pre->>Pre: resize_image (224x224)
    Pre->>Pre: normalize_image (ImageNet mean/std)
    Pre-->>App: original, resized, tensor
    App-->>User: show original and processed image
    App->>Model: load_trained_model() (cached after first call)
    Model-->>App: model, class_names
    App->>Pred: predict(model, tensor, class_names)
    Pred->>Pred: forward pass, softmax, topk
    Pred-->>App: PredictionResult
    App-->>User: crop, condition, status, confidence, top 3
    App->>Ana: make_history_entry(name, result)
    Ana-->>App: history row
    App->>App: session_state.history.append(row)
    User->>App: open Session analytics tab
    App->>Ana: summarize_history, class_distribution, build_text_report
    Ana-->>App: summary, counts, report text
    App-->>User: table, metrics, charts, download buttons
```

## 6. Class / Component Diagram

```mermaid
classDiagram
    class config {
        +IMAGE_SIZE : int
        +IMAGENET_MEAN : list
        +IMAGENET_STD : list
        +ALLOWED_EXTENSIONS : set
        +WEIGHTS_PATH : Path
        +CLASS_NAMES_PATH : Path
        +LOW_CONFIDENCE_THRESHOLD : float
    }

    class preprocessing {
        +validate_upload(file_name, size)
        +decode_image(bytes) ndarray
        +load_image(path) ndarray
        +resize_image(image, size) ndarray
        +normalize_image(image) ndarray
        +to_tensor(image) Tensor
        +preprocess_image(image) Tensor
        +preprocess_upload(name, bytes)
    }
    class ImageValidationError {
        <<exception>>
    }

    class model {
        +build_model(num_classes, pretrained) Module
        +save_model(model, path)
        +load_trained_model(weights, classes, device)
    }

    class prediction {
        +predict(model, tensor, class_names, device, top_k) PredictionResult
    }
    class PredictionResult {
        +class_name : str
        +display_name : str
        +crop : str
        +condition : str
        +confidence : float
        +is_healthy : bool
        +top_k : list
    }

    class analytics {
        +make_history_entry(file_name, result) dict
        +history_to_dataframe(history) DataFrame
        +summarize_history(history) dict
        +class_distribution(history) Series
        +build_text_report(history) str
    }

    class dataset {
        +list_dataset_images(dir) DataFrame
        +create_splits(dir, split_dir) dict
        +load_split(name) DataFrame
    }
    class LeafDataset {
        +table : DataFrame
        +augment : bool
        +__len__()
        +__getitem__(index)
    }

    class evaluation {
        +collect_predictions(model, loader, device)
        +compute_metrics(y_true, y_pred, class_names) dict
        +save_confusion_matrix(y_true, y_pred, class_names, png, csv)
    }

    class utils {
        +set_seed(seed)
        +get_device() device
        +split_class_name(name) tuple
        +is_healthy_class(name) bool
        +format_class_name(name) str
        +save_json(data, path)
        +load_json(path)
    }

    class app {
        +get_model()
        +detection_tab()
        +analytics_tab()
        +evaluation_tab()
        +main()
    }

    preprocessing ..> ImageValidationError : raises
    preprocessing ..> config
    prediction ..> PredictionResult : returns
    prediction ..> utils
    analytics ..> PredictionResult
    dataset ..> preprocessing
    dataset --> LeafDataset
    model ..> config
    evaluation ..> utils
    app ..> preprocessing
    app ..> model
    app ..> prediction
    app ..> analytics
```
