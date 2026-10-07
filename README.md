# Peripheral Blood Smear Transformer

A transformer model that combines the cells of a peripheral blood smear into a slide-level prediction.

- Each cell feature is projected into a token embedding by a linear layer, and its predicted cell type is mapped to a learnable type embedding; the two are added to form the cell token.
- A type-aware Transformer encoder processes all cell tokens of a slide, using padding and attention masking for slides with different numbers of cells.
- Attention pooling aggregates the resulting tokens into a slide-level representation.
- A task-specific classification head produces the outcome prediction (binary or multiclass). 
