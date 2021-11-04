import scipy
from sklearn.metrics.pairwise import cosine_similarity


def test_spearmanr(vid_embedding, annotation_file):
    relevances, similarities = [], []
    with open(annotation_file, 'r') as f:
        for line in f:
            query, candidate, relevance = line.split()
            if query not in vid_embedding:
                raise Exception(f'ERROR: {query} NOT found')
            if candidate not in vid_embedding:
                raise Exception(f'ERROR: {candidate} NOT found')

            query_embedding = vid_embedding.get(query)
            candidate_embedding = vid_embedding.get(candidate)
            similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][0]
            similarities.append(similarity)
            relevances.append(float(relevance))

    spearmanr = scipy.stats.spearmanr(similarities, relevances).correlation
    return spearmanr
