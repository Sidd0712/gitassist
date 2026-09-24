/* API response types mirroring the backend models. */

export interface AmbiguityFlag {
  axis: string;
  question: string;
  reason: string;
  options: string[];
  severity: 'low' | 'medium' | 'high';
  resolved: boolean;
  answer: string | null;
}

export interface ExtractedKeywords {
  keywords: string[];
  frameworks: string[];
  languages: string[];
  summary: string;
  core_intent: string;
  product_type: string;
  target_users: string[];
  capabilities: string[];
  primary_capabilities: string[];
  secondary_capabilities: string[];
  domain_terms: string[];
  tech_terms: string[];
  constraints: string[];
  likely_components: string[];
  likely_integrations: string[];
  likely_stack_families: string[];
  ambiguities: AmbiguityFlag[];
  assumptions: string[];
}

export interface RepoSearchResult {
  full_name: string;
  description: string | null;
  html_url: string;
  stars: number;
  language: string | null;
  topics: string[];
  default_branch?: string;
  updated_at?: string | null;
  archived?: boolean;
  commit_sha?: string;
  relevance_score?: number;
  semantic_meta_score?: number;
  semantic_readme_score?: number;
  query_hit_count?: number;
  rank_reasons?: string[];
  reference_type?: 'candidate' | 'end_to_end' | 'subsystem' | 'pattern';
  fit_score?: number;
  fit_summary?: string;
  covered_primary?: string[];
  missing_primary?: string[];
  evidence_type?: 'deep_retrieval' | 'shallow_evidence' | 'no_evidence';
  dependencies?: string[];
}

export interface LearningStep {
  step_number: number;
  title: string;
  description: string;
  milestone: string;
  concepts: string[];
  resources: string[];
}

export interface TechRecommendation {
  name: string;
  category: string;
  why_recommended: string;
  supported_by: string[];
  pros: string[];
  cons: string[];
}

export interface ClarificationQuestion {
  key: string;
  question: string;
  options: string[];
  reason: string;
}

export interface Citation {
  repo_full_name: string;
  path: string;
  start_line: number | null;
  end_line: number | null;
  reason: string;
  commit_sha?: string | null;
}

export type RepoIndexState = 'not_queued' | 'queued' | 'indexing' | 'partial' | 'completed' | 'failed';

export interface RepoIndexStatus {
  full_name: string;
  state: RepoIndexState;
  chunk_count: number;
  error: string | null;
}

export interface IndexStatusResponse {
  worker_online: boolean;
  repositories: RepoIndexStatus[];
}

export interface RepoChatScopeRepository {
  full_name: string;
  commit_sha: string;
}

export interface RepoChatRequest {
  question: string;
  idea_summary: string;
  scope_repositories: RepoChatScopeRepository[];
  messages: Array<{
    role: 'user' | 'assistant';
    content: string;
  }>;
}

export interface RepoChatEvidenceHit {
  repo_full_name: string;
  path: string;
  start_line: number | null;
  end_line: number | null;
  reason: string;
  snippet: string;
  score: number;
}

export interface RepoChatResponse {
  answer: string;
  citations: Citation[];
  evidence_hits: RepoChatEvidenceHit[];
  follow_up_suggestions: string[];
  scoped_repo_count: number;
}

export interface RepoChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  evidence_hits?: RepoChatEvidenceHit[];
  follow_up_suggestions?: string[];
  isIntro?: boolean;
}

export interface AnalysisEvidence {
  repo_descriptions: Citation[];
  learning_path: Citation[];
  architecture_diagram: Citation[];
  tech_stack: Citation[];
}

export interface AnalysisResponse {
  idea_summary: string;
  keywords: ExtractedKeywords;
  clarification_questions: ClarificationQuestion[];
  assumptions: string[];
  repositories: RepoSearchResult[];
  repo_descriptions: string[];
  learning_path: LearningStep[];
  architecture_diagram: string;
  tech_stack: TechRecommendation[];
  evidence?: AnalysisEvidence | null;
  status: 'needs_clarification' | 'complete' | 'error';
  error: string | null;
}
