export interface Product {
  id: string
  code: string
  name: string
  category: 'Laptop' | 'Mobile Phone'
  brand: string
  price: string
  price_value: number
  description: string
  specs: string[]
  display_specs?: string[]
  matching_facts?: string[]
  highlight_facts?: string[]
  image_url: string
  image_source: 'local' | 'mapped' | 'aura' | 'generated'
  source_url?: string
  fetched_at?: string
  price_valid_until?: string
  spec_provenance?: Record<string, unknown>
}

export interface Bundle {
  title: string
  savings: string
  items: Product[]
}

export interface AISource {
  source?: string
  product_code?: string
  [key: string]: unknown
}

export interface ConversationState {
  category: string | null
  budget_target: number | null
  budget_minimum: number | null
  budget_maximum: number | null
  goal: string | null
  use_case: string | null
  active_product_code: string | null
  compared_codes: string[]
  compared_brands: string[]
  candidate_codes: string[]
  preferences: Record<string, number>
  rejected_codes: Record<string, string>
  last_intent: string | null
  last_recommendation_code: string | null
  topic_id: string | null
  updated_at: string | null
  last_query_frame?: Record<string, unknown>
}

export interface RelatedProductContract {
  product_code: string
  name?: string | null
  brand?: string | null
  price_value?: number | null
  display_specs: string[]
  matching_facts: string[]
  highlight_facts: string[]
}

export interface UIActionContract {
  type: string
  product_codes: string[]
  payload?: Record<string, unknown>
}

export interface ChatResponse {
  text: string
  answer_text?: string | null
  response_mode?: string | null
  query_frame?: Record<string, unknown> | null
  related_products?: RelatedProductContract[]
  ui_actions?: UIActionContract[]
  missing_fields?: string[]
  warnings?: string[]
  workflow_status: string
  ai_mode:
    | 'verification_workflow'
    | 'catalog_fallback'
    | 'deterministic_advisor'
    | 'deterministic_policy'
  tools_used: string[]
  sources: AISource[]
  products: Product[]
  suggest_bundle: boolean
  verification: {
    approved: boolean
    critical_issues: number
    reasoning: string
  } | null
  conversation_state: ConversationState
  answer_type: string
  confidence: number
  active_context: {
    category: string | null
    budget_target: number | null
    compared_brands: string[]
    candidate_codes: string[]
    preferences: Record<string, number>
  }
  follow_up_question: string | null
  decision_trace?: Record<string, unknown> | null
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  products?: Product[]
  bundle?: Bundle
  workflowStatus?: string
  aiMode?: ChatResponse['ai_mode']
  sources?: AISource[]
  verification?: ChatResponse['verification']
}
