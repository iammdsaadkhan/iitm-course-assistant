export function normalizeEmbeddingVector(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  if (!Number.isFinite(norm) || norm === 0) {
    throw new Error("The embedding service returned an invalid vector.");
  }
  return vector.map((value) => value / norm);
}

function averageAndNormalize(value: unknown): number[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("The embedding service returned an empty vector.");
  }

  // HF inference may return one pooled vector or a vector for each token.
  let vector: number[];
  if (typeof value[0] === "number") {
    vector = value as number[];
  } else if (Array.isArray(value[0]) && typeof value[0][0] === "number") {
    const tokenVectors = value as number[][];
    const dimension = tokenVectors[0].length;
    vector = Array.from({ length: dimension }, (_, column) =>
      tokenVectors.reduce((sum, row) => sum + (row[column] ?? 0), 0) / tokenVectors.length,
    );
  } else if (Array.isArray(value[0]) && Array.isArray(value[0][0])) {
    // Handle an extra single-input dimension by flattening one level first.
    return averageAndNormalize(value.flat() as unknown);
  } else {
    throw new Error("The embedding service returned an unexpected vector format.");
  }

  if (vector.some((number) => !Number.isFinite(number))) {
    throw new Error("The embedding service returned an invalid vector.");
  }
  return normalizeEmbeddingVector(vector);
}

/** Convert the different feature-extraction shapes returned by HF providers. */
export function readHuggingFaceEmbeddingRows(output: unknown, expectedCount: number): number[][] {
  if (!Array.isArray(output)) {
    throw new Error("The embedding service returned an unexpected response.");
  }

  // A single input may be returned as a bare vector or as token vectors.
  let rows = output;
  if (expectedCount === 1 && typeof output[0] === "number") {
    rows = [output];
  } else if (
    expectedCount === 1 &&
    output.length > 1 &&
    Array.isArray(output[0]) &&
    typeof output[0][0] === "number"
  ) {
    // Wrap token-by-token output as the one input row; averageAndNormalize pools it.
    rows = [output];
  }
  if (rows.length !== expectedCount) {
    throw new Error("The embedding service returned a different number of vectors than expected.");
  }
  return rows.map(averageAndNormalize);
}

export function readHuggingFaceQuestionVector(output: unknown): number[] {
  if (!Array.isArray(output) || output.length === 0) {
    throw new Error("The embedding service returned an empty question vector.");
  }
  if (typeof output[0] === "number") return normalizeEmbeddingVector(output as number[]);

  const first = output[0];
  if (!Array.isArray(first) || first.length === 0) {
    throw new Error("The embedding service returned an unexpected question vector.");
  }
  if (typeof first[0] === "number") {
    // A one-row output is a pooled vector; multiple rows are token vectors.
    const tokenVectors = output as number[][];
    if (tokenVectors.length === 1) return normalizeEmbeddingVector(tokenVectors[0]);
    const dimension = tokenVectors[0].length;
    const mean = Array.from({ length: dimension }, (_, column) =>
      tokenVectors.reduce((sum, row) => sum + (row[column] ?? 0), 0) / tokenVectors.length,
    );
    return normalizeEmbeddingVector(mean);
  }

  // Some providers wrap token vectors in an extra single-input dimension.
  if (Array.isArray(first[0])) {
    const tokenVectors = first as number[][];
    const dimension = tokenVectors[0].length;
    const mean = Array.from({ length: dimension }, (_, column) =>
      tokenVectors.reduce((sum, row) => sum + (row[column] ?? 0), 0) / tokenVectors.length,
    );
    return normalizeEmbeddingVector(mean);
  }
  throw new Error("The embedding service returned an unexpected question vector.");
}
