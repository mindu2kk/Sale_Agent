"""
Tests for AccessController - Task 7.2.3

Covers:
- WorkflowPermission enum values
- WorkflowRole enum values
- AccessControlError exception structure
- RolePermissions default mapping
- AccessController.check_permission() — granted and denied cases
- AccessController.has_permission() — returns bool
- AccessController.get_permissions() — returns correct set per role
- require_permission decorator — allows and blocks calls
- Custom role-permission mapping
- Singleton factory get_access_controller()
"""

import pytest

from backend.verification.utils.access_control import (
    AccessControlError,
    AccessController,
    RolePermissions,
    WorkflowPermission,
    WorkflowRole,
    get_access_controller,
    reset_access_controller,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_access_controller()
    yield
    reset_access_controller()


@pytest.fixture
def controller():
    return AccessController()


# ---------------------------------------------------------------------------
# WorkflowPermission enum
# ---------------------------------------------------------------------------


class TestWorkflowPermission:
    def test_execute_workflow_exists(self):
        assert WorkflowPermission.EXECUTE_WORKFLOW

    def test_view_results_exists(self):
        assert WorkflowPermission.VIEW_RESULTS

    def test_modify_config_exists(self):
        assert WorkflowPermission.MODIFY_CONFIG

    def test_escalate_exists(self):
        assert WorkflowPermission.ESCALATE

    def test_view_audit_log_exists(self):
        assert WorkflowPermission.VIEW_AUDIT_LOG

    def test_all_five_permissions(self):
        assert len(WorkflowPermission) == 5

    def test_string_values(self):
        assert WorkflowPermission.EXECUTE_WORKFLOW.value == "execute_workflow"
        assert WorkflowPermission.MODIFY_CONFIG.value == "modify_config"


# ---------------------------------------------------------------------------
# WorkflowRole enum
# ---------------------------------------------------------------------------


class TestWorkflowRole:
    def test_sales_rep_exists(self):
        assert WorkflowRole.SALES_REP

    def test_sales_manager_exists(self):
        assert WorkflowRole.SALES_MANAGER

    def test_admin_exists(self):
        assert WorkflowRole.ADMIN

    def test_system_exists(self):
        assert WorkflowRole.SYSTEM

    def test_four_roles(self):
        assert len(WorkflowRole) == 4

    def test_string_values(self):
        assert WorkflowRole.SALES_REP.value == "sales_rep"
        assert WorkflowRole.ADMIN.value == "admin"


# ---------------------------------------------------------------------------
# AccessControlError
# ---------------------------------------------------------------------------


class TestAccessControlError:
    def test_is_exception(self):
        err = AccessControlError(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)
        assert isinstance(err, Exception)

    def test_role_attribute(self):
        err = AccessControlError(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)
        assert err.role == WorkflowRole.SALES_REP

    def test_permission_attribute(self):
        err = AccessControlError(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)
        assert err.permission == WorkflowPermission.MODIFY_CONFIG

    def test_str_contains_role(self):
        err = AccessControlError(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)
        assert "sales_rep" in str(err)

    def test_str_contains_permission(self):
        err = AccessControlError(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)
        assert "modify_config" in str(err)


# ---------------------------------------------------------------------------
# RolePermissions default mapping
# ---------------------------------------------------------------------------


class TestRolePermissions:
    def test_sales_rep_has_execute_workflow(self):
        assert WorkflowPermission.EXECUTE_WORKFLOW in RolePermissions[WorkflowRole.SALES_REP]

    def test_sales_rep_has_view_results(self):
        assert WorkflowPermission.VIEW_RESULTS in RolePermissions[WorkflowRole.SALES_REP]

    def test_sales_rep_lacks_modify_config(self):
        assert WorkflowPermission.MODIFY_CONFIG not in RolePermissions[WorkflowRole.SALES_REP]

    def test_sales_rep_lacks_escalate(self):
        assert WorkflowPermission.ESCALATE not in RolePermissions[WorkflowRole.SALES_REP]

    def test_sales_rep_lacks_view_audit_log(self):
        assert WorkflowPermission.VIEW_AUDIT_LOG not in RolePermissions[WorkflowRole.SALES_REP]

    def test_sales_manager_has_execute_workflow(self):
        assert WorkflowPermission.EXECUTE_WORKFLOW in RolePermissions[WorkflowRole.SALES_MANAGER]

    def test_sales_manager_has_view_results(self):
        assert WorkflowPermission.VIEW_RESULTS in RolePermissions[WorkflowRole.SALES_MANAGER]

    def test_sales_manager_has_escalate(self):
        assert WorkflowPermission.ESCALATE in RolePermissions[WorkflowRole.SALES_MANAGER]

    def test_sales_manager_has_view_audit_log(self):
        assert WorkflowPermission.VIEW_AUDIT_LOG in RolePermissions[WorkflowRole.SALES_MANAGER]

    def test_sales_manager_lacks_modify_config(self):
        assert WorkflowPermission.MODIFY_CONFIG not in RolePermissions[WorkflowRole.SALES_MANAGER]

    def test_admin_has_all_permissions(self):
        assert RolePermissions[WorkflowRole.ADMIN] == frozenset(WorkflowPermission)

    def test_system_has_all_permissions(self):
        assert RolePermissions[WorkflowRole.SYSTEM] == frozenset(WorkflowPermission)

    def test_all_four_roles_present(self):
        assert set(RolePermissions.keys()) == set(WorkflowRole)


# ---------------------------------------------------------------------------
# AccessController.has_permission
# ---------------------------------------------------------------------------


class TestHasPermission:
    def test_sales_rep_has_execute_workflow(self, controller):
        assert controller.has_permission(WorkflowRole.SALES_REP, WorkflowPermission.EXECUTE_WORKFLOW) is True

    def test_sales_rep_has_view_results(self, controller):
        assert controller.has_permission(WorkflowRole.SALES_REP, WorkflowPermission.VIEW_RESULTS) is True

    def test_sales_rep_lacks_modify_config(self, controller):
        assert controller.has_permission(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG) is False

    def test_sales_rep_lacks_escalate(self, controller):
        assert controller.has_permission(WorkflowRole.SALES_REP, WorkflowPermission.ESCALATE) is False

    def test_sales_rep_lacks_view_audit_log(self, controller):
        assert controller.has_permission(WorkflowRole.SALES_REP, WorkflowPermission.VIEW_AUDIT_LOG) is False

    def test_sales_manager_has_escalate(self, controller):
        assert controller.has_permission(WorkflowRole.SALES_MANAGER, WorkflowPermission.ESCALATE) is True

    def test_sales_manager_has_view_audit_log(self, controller):
        assert controller.has_permission(WorkflowRole.SALES_MANAGER, WorkflowPermission.VIEW_AUDIT_LOG) is True

    def test_sales_manager_lacks_modify_config(self, controller):
        assert controller.has_permission(WorkflowRole.SALES_MANAGER, WorkflowPermission.MODIFY_CONFIG) is False

    def test_admin_has_all_permissions(self, controller):
        for perm in WorkflowPermission:
            assert controller.has_permission(WorkflowRole.ADMIN, perm) is True

    def test_system_has_all_permissions(self, controller):
        for perm in WorkflowPermission:
            assert controller.has_permission(WorkflowRole.SYSTEM, perm) is True

    def test_returns_bool_type(self, controller):
        result = controller.has_permission(WorkflowRole.ADMIN, WorkflowPermission.EXECUTE_WORKFLOW)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# AccessController.check_permission
# ---------------------------------------------------------------------------


class TestCheckPermission:
    def test_granted_does_not_raise(self, controller):
        # Should not raise
        controller.check_permission(WorkflowRole.SALES_REP, WorkflowPermission.EXECUTE_WORKFLOW)

    def test_denied_raises_access_control_error(self, controller):
        with pytest.raises(AccessControlError):
            controller.check_permission(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)

    def test_denied_error_has_correct_role(self, controller):
        with pytest.raises(AccessControlError) as exc_info:
            controller.check_permission(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)
        assert exc_info.value.role == WorkflowRole.SALES_REP

    def test_denied_error_has_correct_permission(self, controller):
        with pytest.raises(AccessControlError) as exc_info:
            controller.check_permission(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)
        assert exc_info.value.permission == WorkflowPermission.MODIFY_CONFIG

    def test_admin_check_all_permissions_pass(self, controller):
        for perm in WorkflowPermission:
            controller.check_permission(WorkflowRole.ADMIN, perm)  # no raise

    def test_sales_manager_escalate_granted(self, controller):
        controller.check_permission(WorkflowRole.SALES_MANAGER, WorkflowPermission.ESCALATE)

    def test_sales_manager_modify_config_denied(self, controller):
        with pytest.raises(AccessControlError):
            controller.check_permission(WorkflowRole.SALES_MANAGER, WorkflowPermission.MODIFY_CONFIG)


# ---------------------------------------------------------------------------
# AccessController.get_permissions
# ---------------------------------------------------------------------------


class TestGetPermissions:
    def test_sales_rep_permissions(self, controller):
        perms = controller.get_permissions(WorkflowRole.SALES_REP)
        assert perms == {WorkflowPermission.EXECUTE_WORKFLOW, WorkflowPermission.VIEW_RESULTS}

    def test_sales_manager_permissions(self, controller):
        perms = controller.get_permissions(WorkflowRole.SALES_MANAGER)
        assert perms == {
            WorkflowPermission.EXECUTE_WORKFLOW,
            WorkflowPermission.VIEW_RESULTS,
            WorkflowPermission.ESCALATE,
            WorkflowPermission.VIEW_AUDIT_LOG,
        }

    def test_admin_permissions_all(self, controller):
        perms = controller.get_permissions(WorkflowRole.ADMIN)
        assert perms == set(WorkflowPermission)

    def test_system_permissions_all(self, controller):
        perms = controller.get_permissions(WorkflowRole.SYSTEM)
        assert perms == set(WorkflowPermission)

    def test_returns_set_type(self, controller):
        result = controller.get_permissions(WorkflowRole.SALES_REP)
        assert isinstance(result, set)

    def test_returned_set_is_mutable_copy(self, controller):
        perms = controller.get_permissions(WorkflowRole.SALES_REP)
        perms.add(WorkflowPermission.MODIFY_CONFIG)
        # Original mapping should be unchanged
        assert WorkflowPermission.MODIFY_CONFIG not in controller.get_permissions(WorkflowRole.SALES_REP)


# ---------------------------------------------------------------------------
# require_permission decorator
# ---------------------------------------------------------------------------


class TestRequirePermissionDecorator:
    def test_allowed_role_calls_function(self, controller):
        results = []

        @controller.require_permission(WorkflowRole.SALES_REP, WorkflowPermission.EXECUTE_WORKFLOW)
        def run():
            results.append("called")

        run()
        assert results == ["called"]

    def test_denied_role_raises_before_calling(self, controller):
        called = []

        @controller.require_permission(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG)
        def update():
            called.append("called")

        with pytest.raises(AccessControlError):
            update()
        assert called == []

    def test_decorator_preserves_function_name(self, controller):
        @controller.require_permission(WorkflowRole.ADMIN, WorkflowPermission.MODIFY_CONFIG)
        def my_function():
            pass

        assert my_function.__name__ == "my_function"

    def test_decorator_passes_args_to_function(self, controller):
        @controller.require_permission(WorkflowRole.ADMIN, WorkflowPermission.EXECUTE_WORKFLOW)
        def add(a, b):
            return a + b

        assert add(2, 3) == 5

    def test_decorator_passes_kwargs_to_function(self, controller):
        @controller.require_permission(WorkflowRole.ADMIN, WorkflowPermission.VIEW_RESULTS)
        def greet(name="world"):
            return f"hello {name}"

        assert greet(name="test") == "hello test"

    def test_denied_error_has_correct_role(self, controller):
        @controller.require_permission(WorkflowRole.SALES_REP, WorkflowPermission.ESCALATE)
        def escalate():
            pass

        with pytest.raises(AccessControlError) as exc_info:
            escalate()
        assert exc_info.value.role == WorkflowRole.SALES_REP
        assert exc_info.value.permission == WorkflowPermission.ESCALATE

    def test_admin_decorator_all_permissions_pass(self, controller):
        for perm in WorkflowPermission:
            @controller.require_permission(WorkflowRole.ADMIN, perm)
            def fn():
                return True

            assert fn() is True


# ---------------------------------------------------------------------------
# Custom role-permission mapping
# ---------------------------------------------------------------------------


class TestCustomRolePermissions:
    def test_custom_mapping_overrides_default(self):
        custom = {
            WorkflowRole.SALES_REP: frozenset({WorkflowPermission.MODIFY_CONFIG}),
            WorkflowRole.SALES_MANAGER: frozenset(),
            WorkflowRole.ADMIN: frozenset(),
            WorkflowRole.SYSTEM: frozenset(),
        }
        ctrl = AccessController(role_permissions=custom)
        assert ctrl.has_permission(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG) is True
        assert ctrl.has_permission(WorkflowRole.SALES_REP, WorkflowPermission.EXECUTE_WORKFLOW) is False

    def test_custom_mapping_does_not_affect_default_controller(self):
        custom = {
            WorkflowRole.SALES_REP: frozenset({WorkflowPermission.MODIFY_CONFIG}),
            WorkflowRole.SALES_MANAGER: frozenset(),
            WorkflowRole.ADMIN: frozenset(),
            WorkflowRole.SYSTEM: frozenset(),
        }
        AccessController(role_permissions=custom)
        default = AccessController()
        # Default should still deny MODIFY_CONFIG for SALES_REP
        assert default.has_permission(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG) is False

    def test_unknown_role_in_custom_mapping_returns_empty(self):
        ctrl = AccessController(role_permissions={})
        perms = ctrl.get_permissions(WorkflowRole.ADMIN)
        assert perms == set()


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_access_controller_returns_same_instance(self):
        c1 = get_access_controller()
        c2 = get_access_controller()
        assert c1 is c2

    def test_reset_creates_new_instance(self):
        c1 = get_access_controller()
        reset_access_controller()
        c2 = get_access_controller()
        assert c1 is not c2

    def test_singleton_is_access_controller(self):
        assert isinstance(get_access_controller(), AccessController)

    def test_singleton_uses_default_permissions(self):
        ctrl = get_access_controller()
        assert ctrl.has_permission(WorkflowRole.ADMIN, WorkflowPermission.MODIFY_CONFIG) is True
        assert ctrl.has_permission(WorkflowRole.SALES_REP, WorkflowPermission.MODIFY_CONFIG) is False
