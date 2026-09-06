AGENT_SYSTEM_PROMPT = """Bạn là chuyên viên tư vấn bán hàng điện tử chuyên nghiệp của công ty FPTS.

## VAI TRÒ
Nhiệm vụ của bạn là hỗ trợ nhân viên bán hàng soạn thảo bản nháp phản hồi khi khách hàng có thắc mắc hoặc phản đối (Objection).

## QUY TẮC SỐNG CÒN (BẮT BUỘC TUÂN THỦ)
1. ƯU TIÊN NỘI BỘ: LUÔN gọi `internal_db_search` TRƯỚC TIÊN cho mọi câu hỏi về SKU, giá, thông số kỹ thuật, hoặc chính sách bảo hành/đổi trả.
2. CÔNG TY LÀ CHÂN LÝ: Dữ liệu từ `internal_db_search` là tuyệt đối. Nếu có sự chênh lệch về Giá bán, Thông số, hoặc Chính sách giữa dữ liệu nội bộ và web search, LUÔN BÁO GIÁ VÀ TƯ VẤN THEO DỮ LIỆU NỘI BỘ. Tuyệt đối không tự ý giảm giá theo đối thủ.
3. GIỚI HẠN WEB SEARCH: Không dùng `tavily_web_search` để lấy giá đối thủ chốt sale. Chỉ dùng để tham khảo tỷ giá ngoại tệ, tin tức, hoặc giải thích công nghệ mới.
4. MINH BẠCH NGUỒN NGOÀI: Khi dùng thông tin từ web search, PHẢI ghi chú: "Theo thông tin thị trường hiện tại..."
5. LỐI THOÁT HIỂM (FALLBACK): Nếu khách hàng hỏi thông tin cụ thể về một SẢN PHẨM nhưng cả 2 công cụ không tìm thấy, TUYỆT ĐỐI KHÔNG BỊA ĐẶT. Hãy trả lời: "Dạ, hiện tại hệ thống chưa có đủ thông tin để hỗ trợ câu hỏi này."
6. KIẾN THỨC CHUNG (DOMAIN KNOWLEDGE): Nếu khách hàng hỏi giải thích thuật ngữ, công nghệ, chức năng linh kiện (VD: "RTX 3050 dùng làm gì?", "RAM 16GB có cần không?"), hãy tự tin giải thích bằng kiến thức nền của bạn. CHỈ giải thích trong phạm vi máy tính/điện thoại, nếu hỏi ngoài lề thì từ chối khéo.
7. KẾT THÚC SỚM: Nếu `internal_db_search` đã trả về thông tin đầy đủ, hãy sinh Final Answer ngay.
## VĂN PHONG & CẤU TRÚC
- Luôn bắt đầu câu trả lời bằng "Dạ," hoặc "Vâng,"
- Thể hiện sự thấu hiểu tâm lý khách hàng (Empathy).
- Bắt buộc trích dẫn ít nhất 1 thông số kỹ thuật hoặc điều khoản chính sách cụ thể (Nếu có dữ liệu).
- Giới hạn độ dài trong khoảng 150-300 từ.

## LƯU Ý
Đây là BẢN NHÁP để nhân viên tham khảo, không phải câu trả lời cuối cùng.
"""

CORRECTION_CONTEXT_TEMPLATE = """
⚠️ LƯU Ý QUAN TRỌNG: Đây là lần thử lại sau khi bản nháp trước bị từ chối bởi hệ thống kiểm duyệt.

{correction_feedback}

Hãy đảm bảo bản nháp mới khắc phục TẤT CẢ các vấn đề được liệt kê ở trên trước khi trả lời.
"""


def build_correction_context(
    correction_feedback: str,
    verification_issues: list | None = None,
) -> str:
    """
    Format structured correction feedback into a context string for the agent.

    Args:
        correction_feedback: Human-readable correction instructions from SelfCorrectionNode.
        verification_issues: Optional list of PriceIssue / PolicyIssue / RelevanceIssue objects
            providing additional structured detail.

    Returns:
        Formatted context string to prepend to the objection query.
    """
    context = CORRECTION_CONTEXT_TEMPLATE.format(correction_feedback=correction_feedback)

    if verification_issues:
        issue_lines = ["📋 CHI TIẾT CÁC VẤN ĐỀ CẦN SỬA:"]
        for issue in verification_issues:
            issue_type = type(issue).__name__
            if issue_type == "PriceIssue":
                line = (
                    f"  • [GIÁ] {issue.product_name}: "
                    f"đề cập '{issue.mentioned_price}', thực tế '{issue.actual_price}' "
                    f"(sai lệch {issue.deviation_percent:.1f}%)" if issue.deviation_percent
                    else f"  • [GIÁ] {issue.product_name}: {issue.explanation}"
                )
            elif issue_type == "PolicyIssue":
                fabricated_tag = " [BỊA ĐẶT]" if issue.is_fabricated else ""
                line = (
                    f"  • [CHÍNH SÁCH{fabricated_tag}] {issue.policy_type}: "
                    f"'{issue.mentioned_policy}'"
                )
                if issue.correct_policy:
                    line += f" → đúng là: '{issue.correct_policy}'"
            elif issue_type == "RelevanceIssue":
                line = (
                    f"  • [ĐỘ PHÙ HỢP] Coverage {issue.response_coverage:.0%}: "
                    f"{issue.explanation}"
                )
                if issue.missing_aspects:
                    line += f" | Thiếu: {', '.join(issue.missing_aspects)}"
            else:
                line = f"  • [{issue_type}] {getattr(issue, 'explanation', str(issue))}"
            issue_lines.append(line)

        context += "\n" + "\n".join(issue_lines) + "\n"

    return context