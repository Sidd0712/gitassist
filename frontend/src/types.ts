/* API response types mirroring the backend models. */

export interface ExtractedKeywords {
  keywords: string[];
  frameworks: string[];
  languages: string[];
  summary: string;
  product_type: string;
  target_users: string[];
  capabilities: string[];
  constraints: string[];
  likely_components: string[];
  likely_integrations: string[];
  likely_stack_families: string[];
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
  rank_reasons?: string[];
  reference_type?: 'candidate' | 'end_to_end' | 'subsystem' | 'pattern';
  fit_score?: number;
  fit_summary?: string;
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
