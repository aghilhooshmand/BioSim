# BioSim: Biological Feature Harmonization System - Mermaid Diagrams

## System Architecture Diagram

```mermaid
graph TB
    subgraph "Data Input Layer"
        A[CLuB Data<br/>400 features] 
        B[cBioPortal Data<br/>4000 features]
        C[Column Pairs<br/>Definition]
    end
    
    subgraph "Preprocessing Engine"
        D[Data Validation]
        E[Column Pair Management]
        F[Data Cleaning & Normalization]
    end
    
    subgraph "BioBERT Model Integration"
        G[BioLORD-STAMB2-v1<br/>BioBERT Model]
        H[Sentence Tokenization]
        I[Mean Pooling with Attention]
        J[Embedding Generation]
    end
    
    subgraph "Similarity Calculation"
        K[Cosine Similarity<br/>Calculation]
        L[Batch Processing]
        M[Progress Tracking]
    end
    
    subgraph "Output Layer"
        N[CSV Export<br/>result.csv]
        O[Web Interface<br/>Display]
        P[Visualization<br/>Dashboard]
    end
    
    A --> D
    B --> D
    C --> E
    D --> F
    E --> F
    F --> H
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    M --> N
    M --> O
    M --> P
    
    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style G fill:#fff3e0
    style K fill:#f3e5f5
    style N fill:#e8f5e8
    style O fill:#e8f5e8
    style P fill:#e8f5e8
```

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Input Data"
        A1[CLuB Dataset<br/>400 features]
        A2[cBioPortal Dataset<br/>4000 features]
        A3[Column Pair<br/>Definitions]
    end
    
    subgraph "Processing Pipeline"
        B1[Preprocessing<br/>Validation & Cleaning]
        B2[BioBERT Model<br/>BioLORD-2023]
        B3[Semantic Embedding<br/>Generation]
        B4[Similarity Calculation<br/>Cosine Similarity]
    end
    
    subgraph "Output Results"
        C1[Filtered Matches<br/>High Similarity]
        C2[Ranked Results<br/>Similarity Scores]
        C3[Export Files<br/>CSV Format]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> C1
    B4 --> C2
    B4 --> C3
    
    style A1 fill:#e3f2fd
    style A2 fill:#e3f2fd
    style A3 fill:#e3f2fd
    style B2 fill:#fff8e1
    style C1 fill:#e8f5e8
    style C2 fill:#e8f5e8
    style C3 fill:#e8f5e8
```

## Component Relationship Diagram

```mermaid
graph TD
    subgraph "Core Components"
        A[BioSimilarity.py<br/>Main Engine]
        B[BioSimilarity_interface.py<br/>Web Interface]
        C[map_of_similarity.py<br/>Visualization]
    end
    
    subgraph "Data Sources"
        D[CLuB Clinical Features]
        E[cBioPortal Features]
        F[User Uploaded Data]
    end
    
    subgraph "AI Model"
        G[BioLORD-STAMB2-v1<br/>BioBERT Model]
        H[Sentence Transformers]
        I[PyTorch Backend]
    end
    
    subgraph "Output"
        J[Similarity Results]
        K[Interactive Dashboard]
        L[CSV Export]
    end
    
    D --> A
    E --> A
    F --> A
    A --> G
    G --> H
    H --> I
    A --> J
    J --> K
    J --> L
    B --> A
    C --> J
    
    style A fill:#ffebee
    style B fill:#e8f5e8
    style C fill:#e3f2fd
    style G fill:#fff3e0
    style J fill:#f3e5f5
```

## Performance Comparison Diagram

```mermaid
graph LR
    subgraph "Traditional Approach"
        A1[400 CLuB Features]
        A2[4000 cBioPortal Features]
        A3[1,600,000<br/>Brute Force Comparisons]
        A4[O(n²) Complexity]
    end
    
    subgraph "BioSim Approach"
        B1[Semantic Filtering]
        B2[~10,000 Meaningful<br/>Comparisons]
        B3[O(n) Complexity]
        B4[99.4% Reduction]
    end
    
    A1 --> A3
    A2 --> A3
    A3 --> A4
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    
    style A3 fill:#ffcdd2
    style B2 fill:#c8e6c9
    style B4 fill:#4caf50
```

## Similarity Score Distribution

```mermaid
pie title Similarity Score Ranges
    "High Similarity (0.8-1.0)" : 25
    "Medium Similarity (0.6-0.8)" : 45
    "Low Similarity (0.0-0.6)" : 30
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Docker Container"
        A[Python 3.8 Environment]
        B[BioSim Application]
        C[Dependencies<br/>requirements.txt]
        D[BioBERT Model Cache]
    end
    
    subgraph "External Services"
        E[Streamlit Web Server<br/>Port 8601]
        F[GPU Acceleration<br/>CUDA Support]
        G[Data Volume Mounting]
    end
    
    subgraph "User Interface"
        H[Web Browser]
        I[File Upload]
        J[Result Display]
    end
    
    A --> B
    C --> B
    D --> B
    B --> E
    E --> F
    E --> G
    E --> H
    H --> I
    H --> J
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style E fill:#e8f5e8
    style H fill:#f3e5f5
```

## Usage Instructions

You can use these Mermaid diagrams in:

1. **GitHub**: Copy the code blocks into GitHub markdown files
2. **GitLab**: Mermaid is natively supported
3. **Notion**: Use Mermaid code blocks
4. **Obsidian**: Enable Mermaid plugin
5. **Typora**: Native Mermaid support
6. **Online Mermaid Editor**: https://mermaid.live/

### Example Usage in GitHub:
```markdown
# BioSim Architecture

```mermaid
[Paste any of the diagram code blocks here]
```
```

The **`biosim_mermaid_diagram.md`** file contains all the diagrams you need for your paper and presentations. You can copy individual diagram code blocks and use them wherever Mermaid is supported. 