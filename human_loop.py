class HumanInLoop:

    def review(self, result: dict):
        print("\n===== AI GENERATED RESULT =====")
        print("Prediction:", result["prediction"])
        print("Probability:", result["probability"])
        print("Decision:", result["decision"])
        print("Explanation:\n", result["llm_explanation"])

        print("\n=== Doctor Review Required ===")
        choice = input("Enter decision (approve / reject / modify): ").strip().lower()

        if choice == "approve":
            result["final_decision"] = result["decision"]
            result["doctor_note"] = "Approved by clinician"

        elif choice == "reject":
            result["final_decision"] = "REJECTED → Further investigation required"
            result["doctor_note"] = "Doctor disagrees with AI"

        elif choice == "modify":
            new_decision = input("Enter modified decision: ")
            result["final_decision"] = new_decision
            result["doctor_note"] = "Modified by clinician"

        else:
            result["final_decision"] = result["decision"]
            result["doctor_note"] = "No valid input, default applied"

        return result